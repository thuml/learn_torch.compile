
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


cpp_fused_add_div_embedding_mean_mul_std_sub_unsqueeze_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (128L*x0))];
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 > tmp1;
                    out_ptr0[static_cast<long>(x2 + (128L*x1) + (16384L*x0))] = tmp2;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (128L*x0))];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp7 = in_ptr3[static_cast<long>(x1 + (128L*x0))];
                        auto tmp1 = decltype(tmp0)(tmp0 + 20005);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 20005L), "index out of bounds: 0 <= tmp3 < 20005L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = decltype(tmp7)(tmp7 + 3);
                        auto tmp9 = tmp7 < 0;
                        auto tmp10 = tmp9 ? tmp8 : tmp7;
                        TORCH_CHECK((0 <= tmp10) & (tmp10 < 3L), "index out of bounds: 0 <= tmp10 < 3L")
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (768L*tmp10)));
                        auto tmp12 = tmp6 + tmp11;
                        tmp12.store(out_ptr1 + static_cast<long>(x2 + (768L*x1) + (98304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr3[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr2[static_cast<long>(x0)];
                    auto tmp8 = out_ptr3[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_63 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp0);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = static_cast<float>(768.0);
                    auto tmp4 = tmp2 / tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp1 - tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    auto tmp9 = static_cast<float>(767.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = std::sqrt(tmp10);
                    auto tmp12 = static_cast<float>(1e-06);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 / tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp2);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = static_cast<float>(768.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp3 - tmp7;
                    auto tmp9 = tmp0 * tmp8;
                    auto tmp11 = static_cast<float>(767.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = std::sqrt(tmp12);
                    auto tmp14 = static_cast<float>(1e-06);
                    auto tmp15 = decltype(tmp13)(tmp13 + tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 / tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp4;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp4);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = out_ptr1[static_cast<long>(x0)];
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp5 - tmp9;
                    auto tmp11 = tmp0 * tmp10;
                    auto tmp13 = static_cast<float>(767.0);
                    auto tmp14 = tmp12 / tmp13;
                    auto tmp15 = std::sqrt(tmp14);
                    auto tmp16 = static_cast<float>(1e-06);
                    auto tmp17 = decltype(tmp15)(tmp15 + tmp16);
                    auto tmp18 = at::vec::Vectorized<float>(tmp17);
                    auto tmp19 = tmp11 / tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (768L*x2) + (98304L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (128L*x1) + (128L*x1_inner) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_div_eq_masked_fill_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                                auto tmp4 = in_ptr1[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                                auto tmp1 = c10::convert<long>(tmp0);
                                auto tmp2 = static_cast<long>(0);
                                auto tmp3 = tmp1 == tmp2;
                                auto tmp5 = static_cast<float>(8.0);
                                auto tmp6 = tmp4 / tmp5;
                                auto tmp7 = static_cast<float>(-1000000000.0);
                                auto tmp8 = tmp3 ? tmp7 : tmp6;
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x0))];
                            auto tmp4 = in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (128L*x1) + (1536L*x0))];
                            auto tmp1 = c10::convert<long>(tmp0);
                            auto tmp2 = static_cast<long>(0);
                            auto tmp3 = tmp1 == tmp2;
                            auto tmp5 = static_cast<float>(8.0);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = static_cast<float>(-1000000000.0);
                            auto tmp8 = tmp3 ? tmp7 : tmp6;
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (128L*x2) + (16384L*x1) + (196608L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (8192L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (768L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc1 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        tmp_acc0_vec = tmp_acc0_vec + tmp6;
                        tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp6);
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                    tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = out_ptr0[static_cast<long>(x0)];
                    auto tmp14 = out_ptr1[static_cast<long>(x0)];
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = static_cast<float>(768.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 - tmp11;
                    auto tmp13 = tmp0 * tmp12;
                    auto tmp15 = static_cast<float>(767.0);
                    auto tmp16 = tmp14 / tmp15;
                    auto tmp17 = std::sqrt(tmp16);
                    auto tmp18 = static_cast<float>(1e-06);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = at::vec::Vectorized<float>(tmp19);
                    auto tmp21 = tmp13 / tmp20;
                    auto tmp23 = tmp21 + tmp22;
                    tmp23.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_84 = async_compile.cpp('''
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
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1 = args
    args.clear()
    assert_size_stride(arg0_1, (768, ), (1, ))
    assert_size_stride(arg1_1, (768, ), (1, ))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (768, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (768, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (768, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (768, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (768, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (768, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (768, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (768, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (768, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (20005, 768), (768, 1))
    assert_size_stride(arg49_1, (3, 768), (768, 1))
    assert_size_stride(arg50_1, (768, 768), (768, 1))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, 768), (768, 1))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, 768), (768, 1))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, 768), (768, 1))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (3072, 768), (768, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (768, 3072), (3072, 1))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, 768), (768, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (768, 768), (768, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, 768), (768, 1))
    assert_size_stride(arg67_1, (768, ), (1, ))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (768, 768), (768, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 768), (768, 1))
    assert_size_stride(arg77_1, (768, ), (1, ))
    assert_size_stride(arg78_1, (768, 768), (768, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (768, 768), (768, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (3072, 768), (768, 1))
    assert_size_stride(arg83_1, (3072, ), (1, ))
    assert_size_stride(arg84_1, (768, 3072), (3072, 1))
    assert_size_stride(arg85_1, (768, ), (1, ))
    assert_size_stride(arg86_1, (768, 768), (768, 1))
    assert_size_stride(arg87_1, (768, ), (1, ))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (768, 768), (768, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 768), (768, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (3072, 768), (768, 1))
    assert_size_stride(arg95_1, (3072, ), (1, ))
    assert_size_stride(arg96_1, (768, 3072), (3072, 1))
    assert_size_stride(arg97_1, (768, ), (1, ))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (768, 768), (768, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 768), (768, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (768, 768), (768, 1))
    assert_size_stride(arg105_1, (768, ), (1, ))
    assert_size_stride(arg106_1, (3072, 768), (768, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (768, 3072), (3072, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (768, 768), (768, 1))
    assert_size_stride(arg111_1, (768, ), (1, ))
    assert_size_stride(arg112_1, (768, 768), (768, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (768, 768), (768, 1))
    assert_size_stride(arg115_1, (768, ), (1, ))
    assert_size_stride(arg116_1, (768, 768), (768, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (3072, 768), (768, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (768, 3072), (3072, 1))
    assert_size_stride(arg121_1, (768, ), (1, ))
    assert_size_stride(arg122_1, (768, 768), (768, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (768, 768), (768, 1))
    assert_size_stride(arg125_1, (768, ), (1, ))
    assert_size_stride(arg126_1, (768, 768), (768, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (768, 768), (768, 1))
    assert_size_stride(arg135_1, (768, ), (1, ))
    assert_size_stride(arg136_1, (768, 768), (768, 1))
    assert_size_stride(arg137_1, (768, ), (1, ))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (768, 768), (768, 1))
    assert_size_stride(arg141_1, (768, ), (1, ))
    assert_size_stride(arg142_1, (3072, 768), (768, 1))
    assert_size_stride(arg143_1, (3072, ), (1, ))
    assert_size_stride(arg144_1, (768, 3072), (3072, 1))
    assert_size_stride(arg145_1, (768, ), (1, ))
    assert_size_stride(arg146_1, (768, 768), (768, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (768, 768), (768, 1))
    assert_size_stride(arg151_1, (768, ), (1, ))
    assert_size_stride(arg152_1, (768, 768), (768, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (3072, 768), (768, 1))
    assert_size_stride(arg155_1, (3072, ), (1, ))
    assert_size_stride(arg156_1, (768, 3072), (3072, 1))
    assert_size_stride(arg157_1, (768, ), (1, ))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 768), (768, 1))
    assert_size_stride(arg161_1, (768, ), (1, ))
    assert_size_stride(arg162_1, (768, 768), (768, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (768, 768), (768, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (3072, 768), (768, 1))
    assert_size_stride(arg167_1, (3072, ), (1, ))
    assert_size_stride(arg168_1, (768, 3072), (3072, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 768), (768, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (768, 768), (768, 1))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (768, 768), (768, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (3072, 768), (768, 1))
    assert_size_stride(arg179_1, (3072, ), (1, ))
    assert_size_stride(arg180_1, (768, 3072), (3072, 1))
    assert_size_stride(arg181_1, (768, ), (1, ))
    assert_size_stride(arg182_1, (768, 768), (768, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, 768), (768, 1))
    assert_size_stride(arg185_1, (768, ), (1, ))
    assert_size_stride(arg186_1, (768, 768), (768, 1))
    assert_size_stride(arg187_1, (768, ), (1, ))
    assert_size_stride(arg188_1, (768, 768), (768, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (3072, 768), (768, 1))
    assert_size_stride(arg191_1, (3072, ), (1, ))
    assert_size_stride(arg192_1, (768, 3072), (3072, 1))
    assert_size_stride(arg193_1, (768, ), (1, ))
    assert_size_stride(arg194_1, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(arg195_1, (4, 128), (128, 1))
    assert_size_stride(arg196_1, (4, 128), (128, 1))
    buf0 = empty((4, 1, 128, 128), device='cpu', dtype=torch.bool)
    buf1 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf6 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_mean_mul_std_sub_unsqueeze_0(c_void_p(arg195_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg0_1
    del arg194_1
    del arg195_1
    del arg196_1
    del arg1_1
    del arg48_1
    del arg49_1
    buf7 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg50_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf7)
    del arg50_1
    del arg51_1
    buf8 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg53_1, reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg52_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf8)
    del arg52_1
    del arg53_1
    buf9 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf10 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf9, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf10, (48, 64, 128), (8192, 128, 1), 0), out=buf11)
    buf12 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf11, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf11  # reuse
    buf14 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_div_eq_masked_fill_2(c_void_p(buf13.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf14.data_ptr()))
    buf15 = reinterpret_tensor(buf9, (512, 768), (768, 1), 0); del buf9  # reuse
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg55_1, reinterpret_tensor(buf6, (512, 768), (768, 1), 0), reinterpret_tensor(arg54_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf15)
    del arg54_1
    del arg55_1
    buf16 = buf13; del buf13  # reuse
    buf17 = reinterpret_tensor(buf6, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf6  # reuse
    cpp_fused__softmax_clone_3(c_void_p(buf16.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()))
    buf18 = reinterpret_tensor(buf15, (48, 128, 64), (8192, 64, 1), 0); del buf15  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf16, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf17, (48, 128, 64), (8192, 64, 1), 0), out=buf18)
    buf19 = reinterpret_tensor(buf17, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf17  # reuse
    cpp_fused_clone_4(c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    buf20 = reinterpret_tensor(buf18, (512, 768), (768, 1), 0); del buf18  # reuse
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg57_1, reinterpret_tensor(buf19, (512, 768), (768, 1), 0), reinterpret_tensor(arg56_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf20)
    del arg56_1
    del arg57_1
    buf21 = buf4; del buf4  # reuse
    buf23 = buf2; del buf2  # reuse
    buf25 = reinterpret_tensor(buf19, (4, 128, 768), (98304, 768, 1), 0); del buf19  # reuse
    cpp_fused_add_div_mean_mul_std_sub_5(c_void_p(buf1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg2_1
    del arg3_1
    buf26 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg59_1, reinterpret_tensor(buf25, (512, 768), (768, 1), 0), reinterpret_tensor(arg58_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf26)
    del arg58_1
    del arg59_1
    buf27 = reinterpret_tensor(buf26, (4, 128, 3072), (393216, 3072, 1), 0); del buf26  # reuse
    cpp_fused_gelu_6(c_void_p(buf27.data_ptr()))
    buf28 = reinterpret_tensor(buf25, (512, 768), (768, 1), 0); del buf25  # reuse
    # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg61_1, reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg60_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf28)
    del arg60_1
    del arg61_1
    buf29 = buf23; del buf23  # reuse
    buf31 = buf21; del buf21  # reuse
    buf33 = reinterpret_tensor(buf10, (4, 128, 768), (98304, 768, 1), 0); del buf10  # reuse
    cpp_fused_add_div_mean_mul_std_sub_7(c_void_p(buf1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg4_1
    del arg5_1
    buf34 = buf8; del buf8  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg63_1, reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg62_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf34)
    del arg62_1
    del arg63_1
    buf35 = buf7; del buf7  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg65_1, reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf35)
    del arg64_1
    del arg65_1
    buf36 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf37 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_8(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = reinterpret_tensor(buf16, (48, 128, 128), (16384, 128, 1), 0); del buf16  # reuse
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf36, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf37, (48, 64, 128), (8192, 128, 1), 0), out=buf38)
    buf39 = buf14; del buf14  # reuse
    buf40 = reinterpret_tensor(buf38, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf38  # reuse
    buf41 = buf12; del buf12  # reuse
    cpp_fused__softmax_div_eq_masked_fill_9(c_void_p(buf40.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf37, (512, 768), (768, 1), 0); del buf37  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg67_1, reinterpret_tensor(buf33, (512, 768), (768, 1), 0), reinterpret_tensor(arg66_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
    del arg66_1
    del arg67_1
    buf43 = buf40; del buf40  # reuse
    buf44 = reinterpret_tensor(buf33, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf33  # reuse
    cpp_fused__softmax_clone_10(c_void_p(buf43.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = reinterpret_tensor(buf42, (48, 128, 64), (8192, 64, 1), 0); del buf42  # reuse
    # Source Nodes: [x_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf43, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf44, (48, 128, 64), (8192, 64, 1), 0), out=buf45)
    buf46 = reinterpret_tensor(buf44, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf44  # reuse
    cpp_fused_clone_11(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    buf47 = reinterpret_tensor(buf45, (512, 768), (768, 1), 0); del buf45  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf46, (512, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf47)
    del arg68_1
    del arg69_1
    buf48 = buf31; del buf31  # reuse
    buf50 = buf29; del buf29  # reuse
    buf52 = reinterpret_tensor(buf46, (4, 128, 768), (98304, 768, 1), 0); del buf46  # reuse
    cpp_fused_add_div_mean_mul_std_sub_12(c_void_p(buf1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg6_1
    del arg7_1
    buf53 = reinterpret_tensor(buf27, (512, 3072), (3072, 1), 0); del buf27  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf52, (512, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf53)
    del arg70_1
    del arg71_1
    buf54 = reinterpret_tensor(buf53, (4, 128, 3072), (393216, 3072, 1), 0); del buf53  # reuse
    cpp_fused_gelu_13(c_void_p(buf54.data_ptr()))
    buf55 = reinterpret_tensor(buf52, (512, 768), (768, 1), 0); del buf52  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf55)
    del arg72_1
    del arg73_1
    buf56 = reinterpret_tensor(buf55, (4, 128, 768), (98304, 768, 1), 0); del buf55  # reuse
    buf57 = buf50; del buf50  # reuse
    buf59 = buf48; del buf48  # reuse
    buf61 = reinterpret_tensor(buf36, (4, 128, 768), (98304, 768, 1), 0); del buf36  # reuse
    cpp_fused_add_div_mean_mul_std_sub_14(c_void_p(buf56.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf61.data_ptr()))
    del arg8_1
    del arg9_1
    buf62 = buf47; del buf47  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf62)
    del arg74_1
    del arg75_1
    buf63 = buf28; del buf28  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg76_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf63)
    del arg76_1
    del arg77_1
    buf64 = reinterpret_tensor(buf20, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf20  # reuse
    buf65 = reinterpret_tensor(buf1, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf1  # reuse
    cpp_fused_clone_15(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    buf66 = reinterpret_tensor(buf43, (48, 128, 128), (16384, 128, 1), 0); del buf43  # reuse
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf64, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf65, (48, 64, 128), (8192, 128, 1), 0), out=buf66)
    buf67 = buf41; del buf41  # reuse
    buf68 = reinterpret_tensor(buf66, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf66  # reuse
    buf69 = buf39; del buf39  # reuse
    cpp_fused__softmax_div_eq_masked_fill_16(c_void_p(buf68.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()))
    buf70 = reinterpret_tensor(buf65, (512, 768), (768, 1), 0); del buf65  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf61, (512, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf70)
    del arg78_1
    del arg79_1
    buf71 = buf68; del buf68  # reuse
    buf72 = reinterpret_tensor(buf61, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf61  # reuse
    cpp_fused__softmax_clone_17(c_void_p(buf71.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = reinterpret_tensor(buf70, (48, 128, 64), (8192, 64, 1), 0); del buf70  # reuse
    # Source Nodes: [x_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf71, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf72, (48, 128, 64), (8192, 64, 1), 0), out=buf73)
    buf74 = reinterpret_tensor(buf72, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf72  # reuse
    cpp_fused_clone_18(c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf73, (512, 768), (768, 1), 0); del buf73  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf74, (512, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf75)
    del arg80_1
    del arg81_1
    buf76 = buf59; del buf59  # reuse
    buf78 = buf57; del buf57  # reuse
    buf80 = reinterpret_tensor(buf74, (4, 128, 768), (98304, 768, 1), 0); del buf74  # reuse
    cpp_fused_add_div_mean_mul_std_sub_19(c_void_p(buf56.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg10_1
    del arg11_1
    buf81 = reinterpret_tensor(buf54, (512, 3072), (3072, 1), 0); del buf54  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf80, (512, 768), (768, 1), 0), reinterpret_tensor(arg82_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf81)
    del arg82_1
    del arg83_1
    buf82 = reinterpret_tensor(buf81, (4, 128, 3072), (393216, 3072, 1), 0); del buf81  # reuse
    cpp_fused_gelu_20(c_void_p(buf82.data_ptr()))
    buf83 = reinterpret_tensor(buf80, (512, 768), (768, 1), 0); del buf80  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg85_1, reinterpret_tensor(buf82, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf83)
    del arg84_1
    del arg85_1
    buf84 = buf78; del buf78  # reuse
    buf86 = buf76; del buf76  # reuse
    buf88 = reinterpret_tensor(buf64, (4, 128, 768), (98304, 768, 1), 0); del buf64  # reuse
    cpp_fused_add_div_mean_mul_std_sub_21(c_void_p(buf56.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg12_1
    del arg13_1
    buf89 = buf63; del buf63  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg87_1, reinterpret_tensor(buf88, (512, 768), (768, 1), 0), reinterpret_tensor(arg86_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf89)
    del arg86_1
    del arg87_1
    buf90 = buf62; del buf62  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf88, (512, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf90)
    del arg88_1
    del arg89_1
    buf91 = reinterpret_tensor(buf35, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf35  # reuse
    buf92 = reinterpret_tensor(buf34, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf34  # reuse
    cpp_fused_clone_22(c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = reinterpret_tensor(buf71, (48, 128, 128), (16384, 128, 1), 0); del buf71  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf91, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf92, (48, 64, 128), (8192, 128, 1), 0), out=buf93)
    buf94 = buf69; del buf69  # reuse
    buf95 = reinterpret_tensor(buf93, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf93  # reuse
    buf96 = buf67; del buf67  # reuse
    cpp_fused__softmax_div_eq_masked_fill_23(c_void_p(buf95.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = reinterpret_tensor(buf92, (512, 768), (768, 1), 0); del buf92  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf88, (512, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg90_1
    del arg91_1
    buf98 = buf95; del buf95  # reuse
    buf99 = reinterpret_tensor(buf88, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf88  # reuse
    cpp_fused__softmax_clone_24(c_void_p(buf98.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()))
    buf100 = reinterpret_tensor(buf97, (48, 128, 64), (8192, 64, 1), 0); del buf97  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf98, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf99, (48, 128, 64), (8192, 64, 1), 0), out=buf100)
    buf101 = reinterpret_tensor(buf99, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf99  # reuse
    cpp_fused_clone_25(c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    buf102 = reinterpret_tensor(buf100, (512, 768), (768, 1), 0); del buf100  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf101, (512, 768), (768, 1), 0), reinterpret_tensor(arg92_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf102)
    del arg92_1
    del arg93_1
    buf103 = buf86; del buf86  # reuse
    buf105 = buf84; del buf84  # reuse
    buf107 = reinterpret_tensor(buf101, (4, 128, 768), (98304, 768, 1), 0); del buf101  # reuse
    cpp_fused_add_div_mean_mul_std_sub_26(c_void_p(buf56.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()))
    del arg14_1
    del arg15_1
    buf108 = reinterpret_tensor(buf82, (512, 3072), (3072, 1), 0); del buf82  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg95_1, reinterpret_tensor(buf107, (512, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf108)
    del arg94_1
    del arg95_1
    buf109 = reinterpret_tensor(buf108, (4, 128, 3072), (393216, 3072, 1), 0); del buf108  # reuse
    cpp_fused_gelu_27(c_void_p(buf109.data_ptr()))
    buf110 = reinterpret_tensor(buf107, (512, 768), (768, 1), 0); del buf107  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf109, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf110)
    del arg96_1
    del arg97_1
    buf111 = reinterpret_tensor(buf110, (4, 128, 768), (98304, 768, 1), 0); del buf110  # reuse
    buf112 = buf105; del buf105  # reuse
    buf114 = buf103; del buf103  # reuse
    buf116 = reinterpret_tensor(buf91, (4, 128, 768), (98304, 768, 1), 0); del buf91  # reuse
    cpp_fused_add_div_mean_mul_std_sub_28(c_void_p(buf111.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()))
    del arg16_1
    del arg17_1
    buf117 = buf83; del buf83  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf116, (512, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf117)
    del arg98_1
    del arg99_1
    buf118 = buf75; del buf75  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf116, (512, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf118)
    del arg100_1
    del arg101_1
    buf119 = reinterpret_tensor(buf56, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf56  # reuse
    buf120 = reinterpret_tensor(buf102, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf102  # reuse
    cpp_fused_clone_29(c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()))
    buf121 = reinterpret_tensor(buf98, (48, 128, 128), (16384, 128, 1), 0); del buf98  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf119, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf120, (48, 64, 128), (8192, 128, 1), 0), out=buf121)
    buf122 = buf96; del buf96  # reuse
    buf123 = reinterpret_tensor(buf121, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf121  # reuse
    buf124 = buf94; del buf94  # reuse
    cpp_fused__softmax_div_eq_masked_fill_30(c_void_p(buf123.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = reinterpret_tensor(buf120, (512, 768), (768, 1), 0); del buf120  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf116, (512, 768), (768, 1), 0), reinterpret_tensor(arg102_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf125)
    del arg102_1
    del arg103_1
    buf126 = buf123; del buf123  # reuse
    buf127 = reinterpret_tensor(buf116, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf116  # reuse
    cpp_fused__softmax_clone_31(c_void_p(buf126.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()))
    buf128 = reinterpret_tensor(buf125, (48, 128, 64), (8192, 64, 1), 0); del buf125  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf126, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf127, (48, 128, 64), (8192, 64, 1), 0), out=buf128)
    buf129 = reinterpret_tensor(buf127, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf127  # reuse
    cpp_fused_clone_32(c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()))
    buf130 = reinterpret_tensor(buf128, (512, 768), (768, 1), 0); del buf128  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg105_1, reinterpret_tensor(buf129, (512, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf130)
    del arg104_1
    del arg105_1
    buf131 = buf114; del buf114  # reuse
    buf133 = buf112; del buf112  # reuse
    buf135 = reinterpret_tensor(buf129, (4, 128, 768), (98304, 768, 1), 0); del buf129  # reuse
    cpp_fused_add_div_mean_mul_std_sub_33(c_void_p(buf111.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg18_1
    del arg19_1
    buf136 = reinterpret_tensor(buf109, (512, 3072), (3072, 1), 0); del buf109  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg107_1, reinterpret_tensor(buf135, (512, 768), (768, 1), 0), reinterpret_tensor(arg106_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf136)
    del arg106_1
    del arg107_1
    buf137 = reinterpret_tensor(buf136, (4, 128, 3072), (393216, 3072, 1), 0); del buf136  # reuse
    cpp_fused_gelu_34(c_void_p(buf137.data_ptr()))
    buf138 = reinterpret_tensor(buf135, (512, 768), (768, 1), 0); del buf135  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf137, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg108_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf138)
    del arg108_1
    del arg109_1
    buf139 = buf133; del buf133  # reuse
    buf141 = buf131; del buf131  # reuse
    buf143 = reinterpret_tensor(buf119, (4, 128, 768), (98304, 768, 1), 0); del buf119  # reuse
    cpp_fused_add_div_mean_mul_std_sub_35(c_void_p(buf111.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del arg20_1
    del arg21_1
    buf144 = buf118; del buf118  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf143, (512, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf144)
    del arg110_1
    del arg111_1
    buf145 = buf117; del buf117  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf143, (512, 768), (768, 1), 0), reinterpret_tensor(arg112_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf145)
    del arg112_1
    del arg113_1
    buf146 = reinterpret_tensor(buf90, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf90  # reuse
    buf147 = reinterpret_tensor(buf89, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf89  # reuse
    cpp_fused_clone_36(c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    buf148 = reinterpret_tensor(buf126, (48, 128, 128), (16384, 128, 1), 0); del buf126  # reuse
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf146, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf147, (48, 64, 128), (8192, 128, 1), 0), out=buf148)
    buf149 = buf124; del buf124  # reuse
    buf150 = reinterpret_tensor(buf148, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf148  # reuse
    buf151 = buf122; del buf122  # reuse
    cpp_fused__softmax_div_eq_masked_fill_37(c_void_p(buf150.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = reinterpret_tensor(buf147, (512, 768), (768, 1), 0); del buf147  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf143, (512, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf152)
    del arg114_1
    del arg115_1
    buf153 = buf150; del buf150  # reuse
    buf154 = reinterpret_tensor(buf143, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf143  # reuse
    cpp_fused__softmax_clone_38(c_void_p(buf153.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf152, (48, 128, 64), (8192, 64, 1), 0); del buf152  # reuse
    # Source Nodes: [x_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf153, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf154, (48, 128, 64), (8192, 64, 1), 0), out=buf155)
    buf156 = reinterpret_tensor(buf154, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf154  # reuse
    cpp_fused_clone_39(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf155, (512, 768), (768, 1), 0); del buf155  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg117_1, reinterpret_tensor(buf156, (512, 768), (768, 1), 0), reinterpret_tensor(arg116_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf157)
    del arg116_1
    del arg117_1
    buf158 = buf141; del buf141  # reuse
    buf160 = buf139; del buf139  # reuse
    buf162 = reinterpret_tensor(buf156, (4, 128, 768), (98304, 768, 1), 0); del buf156  # reuse
    cpp_fused_add_div_mean_mul_std_sub_40(c_void_p(buf111.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()))
    del arg22_1
    del arg23_1
    buf163 = reinterpret_tensor(buf137, (512, 3072), (3072, 1), 0); del buf137  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf162, (512, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf163)
    del arg118_1
    del arg119_1
    buf164 = reinterpret_tensor(buf163, (4, 128, 3072), (393216, 3072, 1), 0); del buf163  # reuse
    cpp_fused_gelu_41(c_void_p(buf164.data_ptr()))
    buf165 = reinterpret_tensor(buf162, (512, 768), (768, 1), 0); del buf162  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf164, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg120_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf165)
    del arg120_1
    del arg121_1
    buf166 = reinterpret_tensor(buf165, (4, 128, 768), (98304, 768, 1), 0); del buf165  # reuse
    buf167 = buf160; del buf160  # reuse
    buf169 = buf158; del buf158  # reuse
    buf171 = reinterpret_tensor(buf146, (4, 128, 768), (98304, 768, 1), 0); del buf146  # reuse
    cpp_fused_add_div_mean_mul_std_sub_42(c_void_p(buf166.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()))
    del arg24_1
    del arg25_1
    buf172 = buf157; del buf157  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf171, (512, 768), (768, 1), 0), reinterpret_tensor(arg122_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf172)
    del arg122_1
    del arg123_1
    buf173 = buf138; del buf138  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg125_1, reinterpret_tensor(buf171, (512, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf173)
    del arg124_1
    del arg125_1
    buf174 = reinterpret_tensor(buf130, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf130  # reuse
    buf175 = reinterpret_tensor(buf111, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf111  # reuse
    cpp_fused_clone_43(c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = reinterpret_tensor(buf153, (48, 128, 128), (16384, 128, 1), 0); del buf153  # reuse
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf174, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf175, (48, 64, 128), (8192, 128, 1), 0), out=buf176)
    buf177 = buf151; del buf151  # reuse
    buf178 = reinterpret_tensor(buf176, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf176  # reuse
    buf179 = buf149; del buf149  # reuse
    cpp_fused__softmax_div_eq_masked_fill_44(c_void_p(buf178.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()))
    buf180 = reinterpret_tensor(buf175, (512, 768), (768, 1), 0); del buf175  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg127_1, reinterpret_tensor(buf171, (512, 768), (768, 1), 0), reinterpret_tensor(arg126_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf180)
    del arg126_1
    del arg127_1
    buf181 = buf178; del buf178  # reuse
    buf182 = reinterpret_tensor(buf171, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf171  # reuse
    cpp_fused__softmax_clone_45(c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf180, (48, 128, 64), (8192, 64, 1), 0); del buf180  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf181, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf182, (48, 128, 64), (8192, 64, 1), 0), out=buf183)
    buf184 = reinterpret_tensor(buf182, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf182  # reuse
    cpp_fused_clone_46(c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()))
    buf185 = reinterpret_tensor(buf183, (512, 768), (768, 1), 0); del buf183  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf184, (512, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf185)
    del arg128_1
    del arg129_1
    buf186 = buf169; del buf169  # reuse
    buf188 = buf167; del buf167  # reuse
    buf190 = reinterpret_tensor(buf184, (4, 128, 768), (98304, 768, 1), 0); del buf184  # reuse
    cpp_fused_add_div_mean_mul_std_sub_47(c_void_p(buf166.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf190.data_ptr()))
    del arg26_1
    del arg27_1
    buf191 = reinterpret_tensor(buf164, (512, 3072), (3072, 1), 0); del buf164  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf190, (512, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf191)
    del arg130_1
    del arg131_1
    buf192 = reinterpret_tensor(buf191, (4, 128, 3072), (393216, 3072, 1), 0); del buf191  # reuse
    cpp_fused_gelu_48(c_void_p(buf192.data_ptr()))
    buf193 = reinterpret_tensor(buf190, (512, 768), (768, 1), 0); del buf190  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf192, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf193)
    del arg132_1
    del arg133_1
    buf194 = buf188; del buf188  # reuse
    buf196 = buf186; del buf186  # reuse
    buf198 = reinterpret_tensor(buf174, (4, 128, 768), (98304, 768, 1), 0); del buf174  # reuse
    cpp_fused_add_div_mean_mul_std_sub_49(c_void_p(buf166.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf198.data_ptr()))
    del arg28_1
    del arg29_1
    buf199 = buf173; del buf173  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg135_1, reinterpret_tensor(buf198, (512, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf199)
    del arg134_1
    del arg135_1
    buf200 = buf172; del buf172  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf198, (512, 768), (768, 1), 0), reinterpret_tensor(arg136_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf200)
    del arg136_1
    del arg137_1
    buf201 = reinterpret_tensor(buf145, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf145  # reuse
    buf202 = reinterpret_tensor(buf144, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf144  # reuse
    cpp_fused_clone_50(c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = reinterpret_tensor(buf181, (48, 128, 128), (16384, 128, 1), 0); del buf181  # reuse
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf201, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf202, (48, 64, 128), (8192, 128, 1), 0), out=buf203)
    buf204 = buf179; del buf179  # reuse
    buf205 = reinterpret_tensor(buf203, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf203  # reuse
    buf206 = buf177; del buf177  # reuse
    cpp_fused__softmax_div_eq_masked_fill_51(c_void_p(buf205.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()))
    buf207 = reinterpret_tensor(buf202, (512, 768), (768, 1), 0); del buf202  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf198, (512, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf207)
    del arg138_1
    del arg139_1
    buf208 = buf205; del buf205  # reuse
    buf209 = reinterpret_tensor(buf198, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf198  # reuse
    cpp_fused__softmax_clone_52(c_void_p(buf208.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()))
    buf210 = reinterpret_tensor(buf207, (48, 128, 64), (8192, 64, 1), 0); del buf207  # reuse
    # Source Nodes: [x_61], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf209, (48, 128, 64), (8192, 64, 1), 0), out=buf210)
    buf211 = reinterpret_tensor(buf209, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf209  # reuse
    cpp_fused_clone_53(c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    buf212 = reinterpret_tensor(buf210, (512, 768), (768, 1), 0); del buf210  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf211, (512, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf212)
    del arg140_1
    del arg141_1
    buf213 = buf196; del buf196  # reuse
    buf215 = buf194; del buf194  # reuse
    buf217 = reinterpret_tensor(buf211, (4, 128, 768), (98304, 768, 1), 0); del buf211  # reuse
    cpp_fused_add_div_mean_mul_std_sub_54(c_void_p(buf166.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()))
    del arg30_1
    del arg31_1
    buf218 = reinterpret_tensor(buf192, (512, 3072), (3072, 1), 0); del buf192  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf217, (512, 768), (768, 1), 0), reinterpret_tensor(arg142_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf218)
    del arg142_1
    del arg143_1
    buf219 = reinterpret_tensor(buf218, (4, 128, 3072), (393216, 3072, 1), 0); del buf218  # reuse
    cpp_fused_gelu_55(c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf217, (512, 768), (768, 1), 0); del buf217  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg145_1, reinterpret_tensor(buf219, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg144_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf220)
    del arg144_1
    del arg145_1
    buf221 = reinterpret_tensor(buf220, (4, 128, 768), (98304, 768, 1), 0); del buf220  # reuse
    buf222 = buf215; del buf215  # reuse
    buf224 = buf213; del buf213  # reuse
    buf226 = reinterpret_tensor(buf201, (4, 128, 768), (98304, 768, 1), 0); del buf201  # reuse
    cpp_fused_add_div_mean_mul_std_sub_56(c_void_p(buf221.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg32_1
    del arg33_1
    buf227 = buf212; del buf212  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg147_1, reinterpret_tensor(buf226, (512, 768), (768, 1), 0), reinterpret_tensor(arg146_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf227)
    del arg146_1
    del arg147_1
    buf228 = buf193; del buf193  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf226, (512, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf228)
    del arg148_1
    del arg149_1
    buf229 = reinterpret_tensor(buf185, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf185  # reuse
    buf230 = reinterpret_tensor(buf166, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf166  # reuse
    cpp_fused_clone_57(c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf208, (48, 128, 128), (16384, 128, 1), 0); del buf208  # reuse
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf229, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf230, (48, 64, 128), (8192, 128, 1), 0), out=buf231)
    buf232 = buf206; del buf206  # reuse
    buf233 = reinterpret_tensor(buf231, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf231  # reuse
    buf234 = buf204; del buf204  # reuse
    cpp_fused__softmax_div_eq_masked_fill_58(c_void_p(buf233.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = reinterpret_tensor(buf230, (512, 768), (768, 1), 0); del buf230  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf226, (512, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf235)
    del arg150_1
    del arg151_1
    buf236 = buf233; del buf233  # reuse
    buf237 = reinterpret_tensor(buf226, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf226  # reuse
    cpp_fused__softmax_clone_59(c_void_p(buf236.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf235, (48, 128, 64), (8192, 64, 1), 0); del buf235  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf237, (48, 128, 64), (8192, 64, 1), 0), out=buf238)
    buf239 = reinterpret_tensor(buf237, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf237  # reuse
    cpp_fused_clone_60(c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf238, (512, 768), (768, 1), 0); del buf238  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf239, (512, 768), (768, 1), 0), reinterpret_tensor(arg152_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf240)
    del arg152_1
    del arg153_1
    buf241 = buf224; del buf224  # reuse
    buf243 = buf222; del buf222  # reuse
    buf245 = reinterpret_tensor(buf239, (4, 128, 768), (98304, 768, 1), 0); del buf239  # reuse
    cpp_fused_add_div_mean_mul_std_sub_61(c_void_p(buf221.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()))
    del arg34_1
    del arg35_1
    buf246 = reinterpret_tensor(buf219, (512, 3072), (3072, 1), 0); del buf219  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg155_1, reinterpret_tensor(buf245, (512, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf246)
    del arg154_1
    del arg155_1
    buf247 = reinterpret_tensor(buf246, (4, 128, 3072), (393216, 3072, 1), 0); del buf246  # reuse
    cpp_fused_gelu_62(c_void_p(buf247.data_ptr()))
    buf248 = reinterpret_tensor(buf245, (512, 768), (768, 1), 0); del buf245  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg157_1, reinterpret_tensor(buf247, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg156_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf248)
    del arg156_1
    del arg157_1
    buf249 = buf243; del buf243  # reuse
    buf251 = buf241; del buf241  # reuse
    buf253 = reinterpret_tensor(buf229, (4, 128, 768), (98304, 768, 1), 0); del buf229  # reuse
    cpp_fused_add_div_mean_mul_std_sub_63(c_void_p(buf221.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()))
    del arg36_1
    del arg37_1
    buf254 = buf228; del buf228  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf253, (512, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf254)
    del arg158_1
    del arg159_1
    buf255 = buf227; del buf227  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf253, (512, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf255)
    del arg160_1
    del arg161_1
    buf256 = reinterpret_tensor(buf200, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf200  # reuse
    buf257 = reinterpret_tensor(buf199, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf199  # reuse
    cpp_fused_clone_64(c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    buf258 = reinterpret_tensor(buf236, (48, 128, 128), (16384, 128, 1), 0); del buf236  # reuse
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf256, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf257, (48, 64, 128), (8192, 128, 1), 0), out=buf258)
    buf259 = buf234; del buf234  # reuse
    buf260 = reinterpret_tensor(buf258, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf258  # reuse
    buf261 = buf232; del buf232  # reuse
    cpp_fused__softmax_div_eq_masked_fill_65(c_void_p(buf260.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf257, (512, 768), (768, 1), 0); del buf257  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf253, (512, 768), (768, 1), 0), reinterpret_tensor(arg162_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf262)
    del arg162_1
    del arg163_1
    buf263 = buf260; del buf260  # reuse
    buf264 = reinterpret_tensor(buf253, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf253  # reuse
    cpp_fused__softmax_clone_66(c_void_p(buf263.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()))
    buf265 = reinterpret_tensor(buf262, (48, 128, 64), (8192, 64, 1), 0); del buf262  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf263, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf264, (48, 128, 64), (8192, 64, 1), 0), out=buf265)
    buf266 = reinterpret_tensor(buf264, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf264  # reuse
    cpp_fused_clone_67(c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf265, (512, 768), (768, 1), 0); del buf265  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf266, (512, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf267)
    del arg164_1
    del arg165_1
    buf268 = buf251; del buf251  # reuse
    buf270 = buf249; del buf249  # reuse
    buf272 = reinterpret_tensor(buf266, (4, 128, 768), (98304, 768, 1), 0); del buf266  # reuse
    cpp_fused_add_div_mean_mul_std_sub_68(c_void_p(buf221.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf272.data_ptr()))
    del arg38_1
    del arg39_1
    buf273 = reinterpret_tensor(buf247, (512, 3072), (3072, 1), 0); del buf247  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf272, (512, 768), (768, 1), 0), reinterpret_tensor(arg166_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf273)
    del arg166_1
    del arg167_1
    buf274 = reinterpret_tensor(buf273, (4, 128, 3072), (393216, 3072, 1), 0); del buf273  # reuse
    cpp_fused_gelu_69(c_void_p(buf274.data_ptr()))
    buf275 = reinterpret_tensor(buf272, (512, 768), (768, 1), 0); del buf272  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg169_1, reinterpret_tensor(buf274, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg168_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf275)
    del arg168_1
    del arg169_1
    buf276 = reinterpret_tensor(buf275, (4, 128, 768), (98304, 768, 1), 0); del buf275  # reuse
    buf277 = buf270; del buf270  # reuse
    buf279 = buf268; del buf268  # reuse
    buf281 = reinterpret_tensor(buf256, (4, 128, 768), (98304, 768, 1), 0); del buf256  # reuse
    cpp_fused_add_div_mean_mul_std_sub_70(c_void_p(buf276.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg40_1
    del arg41_1
    buf282 = buf267; del buf267  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg171_1, reinterpret_tensor(buf281, (512, 768), (768, 1), 0), reinterpret_tensor(arg170_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf282)
    del arg170_1
    del arg171_1
    buf283 = buf248; del buf248  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg173_1, reinterpret_tensor(buf281, (512, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf283)
    del arg172_1
    del arg173_1
    buf284 = reinterpret_tensor(buf240, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf240  # reuse
    buf285 = reinterpret_tensor(buf221, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf221  # reuse
    cpp_fused_clone_71(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = reinterpret_tensor(buf263, (48, 128, 128), (16384, 128, 1), 0); del buf263  # reuse
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf285, (48, 64, 128), (8192, 128, 1), 0), out=buf286)
    buf287 = buf261; del buf261  # reuse
    buf288 = reinterpret_tensor(buf286, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf286  # reuse
    buf289 = buf259; del buf259  # reuse
    cpp_fused__softmax_div_eq_masked_fill_72(c_void_p(buf288.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = reinterpret_tensor(buf285, (512, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf281, (512, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf290)
    del arg174_1
    del arg175_1
    buf291 = buf288; del buf288  # reuse
    buf292 = reinterpret_tensor(buf281, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf281  # reuse
    cpp_fused__softmax_clone_73(c_void_p(buf291.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf292.data_ptr()))
    buf293 = reinterpret_tensor(buf290, (48, 128, 64), (8192, 64, 1), 0); del buf290  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf291, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf292, (48, 128, 64), (8192, 64, 1), 0), out=buf293)
    buf294 = reinterpret_tensor(buf292, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf292  # reuse
    cpp_fused_clone_74(c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()))
    buf295 = reinterpret_tensor(buf293, (512, 768), (768, 1), 0); del buf293  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf294, (512, 768), (768, 1), 0), reinterpret_tensor(arg176_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf295)
    del arg176_1
    del arg177_1
    buf296 = buf279; del buf279  # reuse
    buf298 = buf277; del buf277  # reuse
    buf300 = reinterpret_tensor(buf294, (4, 128, 768), (98304, 768, 1), 0); del buf294  # reuse
    cpp_fused_add_div_mean_mul_std_sub_75(c_void_p(buf276.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()))
    del arg42_1
    del arg43_1
    buf301 = reinterpret_tensor(buf274, (512, 3072), (3072, 1), 0); del buf274  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg179_1, reinterpret_tensor(buf300, (512, 768), (768, 1), 0), reinterpret_tensor(arg178_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf301)
    del arg178_1
    del arg179_1
    buf302 = reinterpret_tensor(buf301, (4, 128, 3072), (393216, 3072, 1), 0); del buf301  # reuse
    cpp_fused_gelu_76(c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf300, (512, 768), (768, 1), 0); del buf300  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf302, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg180_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf303)
    del arg180_1
    del arg181_1
    buf304 = buf298; del buf298  # reuse
    buf306 = buf296; del buf296  # reuse
    buf308 = reinterpret_tensor(buf284, (4, 128, 768), (98304, 768, 1), 0); del buf284  # reuse
    cpp_fused_add_div_mean_mul_std_sub_77(c_void_p(buf276.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf308.data_ptr()))
    del arg44_1
    del arg45_1
    buf309 = buf283; del buf283  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg183_1, reinterpret_tensor(buf308, (512, 768), (768, 1), 0), reinterpret_tensor(arg182_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf309)
    del arg182_1
    del arg183_1
    buf310 = buf282; del buf282  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf308, (512, 768), (768, 1), 0), reinterpret_tensor(arg184_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf310)
    del arg184_1
    del arg185_1
    buf311 = reinterpret_tensor(buf255, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf255  # reuse
    buf312 = reinterpret_tensor(buf254, (4, 12, 64, 128), (98304, 8192, 128, 1), 0); del buf254  # reuse
    cpp_fused_clone_78(c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del buf309
    del buf310
    buf313 = reinterpret_tensor(buf291, (48, 128, 128), (16384, 128, 1), 0); del buf291  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf311, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf312, (48, 64, 128), (8192, 128, 1), 0), out=buf313)
    del buf311
    buf314 = buf289; del buf289  # reuse
    buf315 = reinterpret_tensor(buf313, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf313  # reuse
    buf316 = buf287; del buf287  # reuse
    cpp_fused__softmax_div_eq_masked_fill_79(c_void_p(buf315.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()))
    del buf314
    buf317 = reinterpret_tensor(buf312, (512, 768), (768, 1), 0); del buf312  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf308, (512, 768), (768, 1), 0), reinterpret_tensor(arg186_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf317)
    del arg186_1
    del arg187_1
    buf318 = buf315; del buf315  # reuse
    buf319 = reinterpret_tensor(buf308, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf308  # reuse
    cpp_fused__softmax_clone_80(c_void_p(buf318.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf319.data_ptr()))
    del buf316
    buf320 = reinterpret_tensor(buf317, (48, 128, 64), (8192, 64, 1), 0); del buf317  # reuse
    # Source Nodes: [x_93], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf318, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf319, (48, 128, 64), (8192, 64, 1), 0), out=buf320)
    del buf318
    buf321 = reinterpret_tensor(buf319, (4, 128, 12, 64), (98304, 768, 64, 1), 0); del buf319  # reuse
    cpp_fused_clone_81(c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    buf322 = reinterpret_tensor(buf320, (512, 768), (768, 1), 0); del buf320  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg189_1, reinterpret_tensor(buf321, (512, 768), (768, 1), 0), reinterpret_tensor(arg188_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf322)
    del arg188_1
    del arg189_1
    buf323 = buf306; del buf306  # reuse
    buf325 = buf304; del buf304  # reuse
    buf327 = reinterpret_tensor(buf321, (4, 128, 768), (98304, 768, 1), 0); del buf321  # reuse
    cpp_fused_add_div_mean_mul_std_sub_82(c_void_p(buf276.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()))
    del arg46_1
    del arg47_1
    del buf323
    del buf325
    buf328 = reinterpret_tensor(buf302, (512, 3072), (3072, 1), 0); del buf302  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg191_1, reinterpret_tensor(buf327, (512, 768), (768, 1), 0), reinterpret_tensor(arg190_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf328)
    del arg190_1
    del arg191_1
    buf329 = reinterpret_tensor(buf328, (4, 128, 3072), (393216, 3072, 1), 0); del buf328  # reuse
    cpp_fused_gelu_83(c_void_p(buf329.data_ptr()))
    buf330 = reinterpret_tensor(buf327, (512, 768), (768, 1), 0); del buf327  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg193_1, reinterpret_tensor(buf329, (512, 3072), (3072, 1), 0), reinterpret_tensor(arg192_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf330)
    del arg192_1
    del arg193_1
    del buf329
    buf331 = reinterpret_tensor(buf330, (4, 128, 768), (98304, 768, 1), 0); del buf330  # reuse
    cpp_fused_add_84(c_void_p(buf331.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf322.data_ptr()))
    return (buf331, buf0, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((20005, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((3, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    arg196_1 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
