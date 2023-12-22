
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


cpp_fused_add_div_embedding_mean_mul_std_sub_unsqueeze_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       bool* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr3 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr5 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused__softmax_clone_div_eq_masked_fill_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_7 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_12 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_13 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_19 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_24 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_25 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_36 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_37 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused__softmax_clone_div_eq_masked_fill_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_48 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_49 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_55 = async_compile.cpp('''
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


cpp_fused__softmax_clone_div_eq_masked_fill_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_60 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr1;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp8 = in_out_ptr1[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp0 - tmp4;
                    auto tmp7 = tmp6 * tmp5;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp7 / tmp11;
                    auto tmp14 = tmp12 + tmp13;
                    tmp5.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp14.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_61 = async_compile.cpp('''
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


cpp_fused__softmax_clone_detach_div_eq_masked_fill_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp10 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp2 - tmp6;
                    auto tmp9 = tmp8 * tmp7;
                    auto tmp11 = static_cast<float>(1e-06);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp9 / tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    tmp7.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused_add_div_mean_mul_std_sub_view_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp12 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp4 - tmp8;
                    auto tmp11 = tmp10 * tmp9;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp11 / tmp15;
                    auto tmp18 = tmp16 + tmp17;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
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


cpp_fused__softmax_clone_detach_div_eq_masked_fill_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const bool* in_ptr0,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    tmp3.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (98304L*x0)));
                            tmp0.store(out_ptr4 + static_cast<long>(x3 + (64L*x2) + (8192L*x1) + (98304L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((64L*(static_cast<long>(x0) % static_cast<long>(128L))) + (8192L*(c10::div_floor_integer((x1 + x1_inner), 64L))) + (98304L*(c10::div_floor_integer(x0, 128L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mean_mul_std_sub_view_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(767.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = tmp3.sqrt();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp6 - tmp10;
                    auto tmp13 = tmp12 * tmp11;
                    auto tmp15 = static_cast<float>(1e-06);
                    auto tmp16 = decltype(tmp14)(tmp14 + tmp15);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = tmp13 / tmp17;
                    auto tmp20 = tmp18 + tmp19;
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp20.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
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


cpp_fused__softmax_add_detach_72 = async_compile.cpp('''
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
                       const float* in_ptr13)
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr4[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr5[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr2 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr6[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr7[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr8[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr5 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr9[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr6 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr10[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr7 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr11[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr8 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr12[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr9 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6144L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = in_ptr13[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr10 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197 = args
    args.clear()
    assert_size_stride(primals_1, (768, ), (1, ))
    assert_size_stride(primals_2, (768, ), (1, ))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (768, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (768, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (768, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (768, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (20005, 768), (768, 1))
    assert_size_stride(primals_50, (3, 768), (768, 1))
    assert_size_stride(primals_51, (768, 768), (768, 1))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, 768), (768, 1))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768), (768, 1))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (3072, 768), (768, 1))
    assert_size_stride(primals_60, (3072, ), (1, ))
    assert_size_stride(primals_61, (768, 3072), (3072, 1))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 768), (768, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768), (768, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, 768), (768, 1))
    assert_size_stride(primals_68, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (768, 768), (768, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 768), (768, 1))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (768, 768), (768, 1))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768), (768, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (3072, 768), (768, 1))
    assert_size_stride(primals_84, (3072, ), (1, ))
    assert_size_stride(primals_85, (768, 3072), (3072, 1))
    assert_size_stride(primals_86, (768, ), (1, ))
    assert_size_stride(primals_87, (768, 768), (768, 1))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (768, 768), (768, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 768), (768, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (3072, 768), (768, 1))
    assert_size_stride(primals_96, (3072, ), (1, ))
    assert_size_stride(primals_97, (768, 3072), (3072, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 768), (768, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, 768), (768, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768), (768, 1))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (3072, 768), (768, 1))
    assert_size_stride(primals_108, (3072, ), (1, ))
    assert_size_stride(primals_109, (768, 3072), (3072, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (768, 768), (768, 1))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768), (768, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (768, 768), (768, 1))
    assert_size_stride(primals_116, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 768), (768, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (3072, 768), (768, 1))
    assert_size_stride(primals_120, (3072, ), (1, ))
    assert_size_stride(primals_121, (768, 3072), (3072, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, 768), (768, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 768), (768, 1))
    assert_size_stride(primals_126, (768, ), (1, ))
    assert_size_stride(primals_127, (768, 768), (768, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, 768), (768, 1))
    assert_size_stride(primals_136, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768), (768, 1))
    assert_size_stride(primals_138, (768, ), (1, ))
    assert_size_stride(primals_139, (768, 768), (768, 1))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 768), (768, 1))
    assert_size_stride(primals_142, (768, ), (1, ))
    assert_size_stride(primals_143, (3072, 768), (768, 1))
    assert_size_stride(primals_144, (3072, ), (1, ))
    assert_size_stride(primals_145, (768, 3072), (3072, 1))
    assert_size_stride(primals_146, (768, ), (1, ))
    assert_size_stride(primals_147, (768, 768), (768, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (768, 768), (768, 1))
    assert_size_stride(primals_152, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768), (768, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (3072, 768), (768, 1))
    assert_size_stride(primals_156, (3072, ), (1, ))
    assert_size_stride(primals_157, (768, 3072), (3072, 1))
    assert_size_stride(primals_158, (768, ), (1, ))
    assert_size_stride(primals_159, (768, 768), (768, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768), (768, 1))
    assert_size_stride(primals_162, (768, ), (1, ))
    assert_size_stride(primals_163, (768, 768), (768, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 768), (768, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (3072, 768), (768, 1))
    assert_size_stride(primals_168, (3072, ), (1, ))
    assert_size_stride(primals_169, (768, 3072), (3072, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, 768), (768, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 768), (768, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (768, 768), (768, 1))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768), (768, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (3072, 768), (768, 1))
    assert_size_stride(primals_180, (3072, ), (1, ))
    assert_size_stride(primals_181, (768, 3072), (3072, 1))
    assert_size_stride(primals_182, (768, ), (1, ))
    assert_size_stride(primals_183, (768, 768), (768, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768), (768, 1))
    assert_size_stride(primals_186, (768, ), (1, ))
    assert_size_stride(primals_187, (768, 768), (768, 1))
    assert_size_stride(primals_188, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 768), (768, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (3072, 768), (768, 1))
    assert_size_stride(primals_192, (3072, ), (1, ))
    assert_size_stride(primals_193, (768, 3072), (3072, 1))
    assert_size_stride(primals_194, (768, ), (1, ))
    assert_size_stride(primals_195, (1, 512, 768), (393216, 768, 1))
    assert_size_stride(primals_196, (4, 128), (128, 1))
    assert_size_stride(primals_197, (4, 128), (128, 1))
    buf0 = empty((4, 1, 128, 128), device='cpu', dtype=torch.bool)
    buf1 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf4, (4, 128, 1), (128, 1, 1), 0); del buf4  # reuse
    buf7 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf8 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_embedding_mean_mul_std_sub_unsqueeze_view_0(c_void_p(buf6.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del primals_195
    del primals_2
    del primals_49
    del primals_50
    buf9 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, buf8, reinterpret_tensor(primals_51, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf9)
    del primals_52
    buf10 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_54, buf8, reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf10)
    del primals_54
    buf11 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_56, buf8, reinterpret_tensor(primals_55, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf11)
    del primals_56
    buf12 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf13 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_1(c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf12, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf13, (48, 64, 128), (8192, 128, 1), 0), out=buf14)
    buf15 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf14, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf14  # reuse
    buf17 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf18 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf19 = reinterpret_tensor(buf9, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf9  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_2(c_void_p(buf16.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    buf20 = reinterpret_tensor(buf11, (48, 128, 64), (8192, 64, 1), 0); del buf11  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf18, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf19, (48, 128, 64), (8192, 64, 1), 0), out=buf20)
    buf21 = buf10; del buf10  # reuse
    cpp_fused_view_3(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = reinterpret_tensor(buf20, (512, 768), (768, 1), 0); del buf20  # reuse
    # Source Nodes: [l__mod___transformer_blocks_0_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_58, buf21, reinterpret_tensor(primals_57, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf22)
    del primals_58
    buf23 = buf2; del buf2  # reuse
    buf25 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf25, (4, 128, 1), (128, 1, 1), 0); del buf25  # reuse
    buf28 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf29 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_4(c_void_p(buf27.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del primals_4
    buf30 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_60, buf29, reinterpret_tensor(primals_59, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf30)
    del primals_60
    buf31 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_5(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    buf32 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_0_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_62, buf31, reinterpret_tensor(primals_61, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf32)
    del primals_62
    buf33 = buf23; del buf23  # reuse
    buf35 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf37 = reinterpret_tensor(buf35, (4, 128, 1), (128, 1, 1), 0); del buf35  # reuse
    buf38 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf39 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_6(c_void_p(buf37.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_6
    buf40 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_64, buf39, reinterpret_tensor(primals_63, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf40)
    del primals_64
    buf41 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_66, buf39, reinterpret_tensor(primals_65, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf41)
    del primals_66
    buf42 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_68, buf39, reinterpret_tensor(primals_67, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf42)
    del primals_68
    buf43 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf44 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_7(c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    buf45 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf43, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf44, (48, 64, 128), (8192, 128, 1), 0), out=buf45)
    buf46 = buf15; del buf15  # reuse
    buf47 = reinterpret_tensor(buf45, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf45  # reuse
    buf48 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf49 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf50 = reinterpret_tensor(buf41, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf41  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_8(c_void_p(buf47.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()))
    buf51 = reinterpret_tensor(buf42, (48, 128, 64), (8192, 64, 1), 0); del buf42  # reuse
    # Source Nodes: [x_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf49, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf50, (48, 128, 64), (8192, 64, 1), 0), out=buf51)
    buf52 = buf40; del buf40  # reuse
    cpp_fused_view_9(c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    buf53 = reinterpret_tensor(buf51, (512, 768), (768, 1), 0); del buf51  # reuse
    # Source Nodes: [l__mod___transformer_blocks_1_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, buf52, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf53)
    del primals_70
    buf54 = buf33; del buf33  # reuse
    buf56 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf58 = reinterpret_tensor(buf56, (4, 128, 1), (128, 1, 1), 0); del buf56  # reuse
    buf59 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf60 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_10(c_void_p(buf58.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    del primals_8
    buf61 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf60, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf61)
    del primals_72
    buf62 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_11(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    buf63 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_1_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf62, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf63)
    del primals_74
    buf64 = reinterpret_tensor(buf63, (4, 128, 768), (98304, 768, 1), 0); del buf63  # reuse
    buf65 = buf54; del buf54  # reuse
    buf67 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf69 = reinterpret_tensor(buf67, (4, 128, 1), (128, 1, 1), 0); del buf67  # reuse
    buf70 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf71 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_12(c_void_p(buf64.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_10
    buf72 = buf53; del buf53  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, buf71, reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf72)
    del primals_76
    buf73 = buf32; del buf32  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf71, reinterpret_tensor(primals_77, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf73)
    del primals_78
    buf74 = buf22; del buf22  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf71, reinterpret_tensor(primals_79, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf74)
    del primals_80
    buf75 = reinterpret_tensor(buf1, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf1  # reuse
    buf76 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_13(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf75, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf76, (48, 64, 128), (8192, 128, 1), 0), out=buf77)
    buf78 = buf46; del buf46  # reuse
    buf79 = reinterpret_tensor(buf77, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf77  # reuse
    buf80 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf81 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf82 = reinterpret_tensor(buf73, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf73  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_14(c_void_p(buf79.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = reinterpret_tensor(buf74, (48, 128, 64), (8192, 64, 1), 0); del buf74  # reuse
    # Source Nodes: [x_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf82, (48, 128, 64), (8192, 64, 1), 0), out=buf83)
    buf84 = buf72; del buf72  # reuse
    cpp_fused_view_15(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    buf85 = reinterpret_tensor(buf83, (512, 768), (768, 1), 0); del buf83  # reuse
    # Source Nodes: [l__mod___transformer_blocks_2_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf84, reinterpret_tensor(primals_81, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf85)
    del primals_82
    buf86 = buf65; del buf65  # reuse
    buf88 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf90 = reinterpret_tensor(buf88, (4, 128, 1), (128, 1, 1), 0); del buf88  # reuse
    buf91 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf92 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_16(c_void_p(buf90.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_12
    buf93 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf92, reinterpret_tensor(primals_83, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf93)
    del primals_84
    buf94 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_17(c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    buf95 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_2_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_86, buf94, reinterpret_tensor(primals_85, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf95)
    del primals_86
    buf96 = buf86; del buf86  # reuse
    buf98 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf100 = reinterpret_tensor(buf98, (4, 128, 1), (128, 1, 1), 0); del buf98  # reuse
    buf101 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf102 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_18(c_void_p(buf100.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_14
    buf103 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_88, buf102, reinterpret_tensor(primals_87, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf103)
    del primals_88
    buf104 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf102, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf104)
    del primals_90
    buf105 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf102, reinterpret_tensor(primals_91, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf105)
    del primals_92
    buf106 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf107 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_19(c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    buf108 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf106, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf107, (48, 64, 128), (8192, 128, 1), 0), out=buf108)
    buf109 = buf78; del buf78  # reuse
    buf110 = reinterpret_tensor(buf108, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf108  # reuse
    buf111 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf112 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf113 = reinterpret_tensor(buf104, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf104  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_20(c_void_p(buf110.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = reinterpret_tensor(buf105, (48, 128, 64), (8192, 64, 1), 0); del buf105  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf113, (48, 128, 64), (8192, 64, 1), 0), out=buf114)
    buf115 = buf103; del buf103  # reuse
    cpp_fused_view_21(c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()))
    buf116 = reinterpret_tensor(buf114, (512, 768), (768, 1), 0); del buf114  # reuse
    # Source Nodes: [l__mod___transformer_blocks_3_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf115, reinterpret_tensor(primals_93, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf116)
    del primals_94
    buf117 = buf96; del buf96  # reuse
    buf119 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf121 = reinterpret_tensor(buf119, (4, 128, 1), (128, 1, 1), 0); del buf119  # reuse
    buf122 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf123 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_22(c_void_p(buf121.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_16
    buf124 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_96, buf123, reinterpret_tensor(primals_95, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf124)
    del primals_96
    buf125 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_23(c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()))
    buf126 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_3_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf125, reinterpret_tensor(primals_97, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf126)
    del primals_98
    buf127 = reinterpret_tensor(buf126, (4, 128, 768), (98304, 768, 1), 0); del buf126  # reuse
    buf128 = buf117; del buf117  # reuse
    buf130 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf130, (4, 128, 1), (128, 1, 1), 0); del buf130  # reuse
    buf133 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf134 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_24(c_void_p(buf127.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    del primals_18
    buf135 = buf95; del buf95  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf134, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf135)
    del primals_100
    buf136 = buf85; del buf85  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf134, reinterpret_tensor(primals_101, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf136)
    del primals_102
    buf137 = reinterpret_tensor(buf64, (512, 768), (768, 1), 0); del buf64  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf134, reinterpret_tensor(primals_103, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf137)
    del primals_104
    buf138 = reinterpret_tensor(buf116, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf116  # reuse
    buf139 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_25(c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf139, (48, 64, 128), (8192, 128, 1), 0), out=buf140)
    buf141 = buf109; del buf109  # reuse
    buf142 = reinterpret_tensor(buf140, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf140  # reuse
    buf143 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf144 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf145 = reinterpret_tensor(buf136, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf136  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_26(c_void_p(buf142.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf137, (48, 128, 64), (8192, 64, 1), 0); del buf137  # reuse
    # Source Nodes: [x_37], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf144, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf145, (48, 128, 64), (8192, 64, 1), 0), out=buf146)
    buf147 = buf135; del buf135  # reuse
    cpp_fused_view_27(c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    buf148 = reinterpret_tensor(buf146, (512, 768), (768, 1), 0); del buf146  # reuse
    # Source Nodes: [l__mod___transformer_blocks_4_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_106, buf147, reinterpret_tensor(primals_105, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf148)
    del primals_106
    buf149 = buf128; del buf128  # reuse
    buf151 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf153 = reinterpret_tensor(buf151, (4, 128, 1), (128, 1, 1), 0); del buf151  # reuse
    buf154 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf155 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_28(c_void_p(buf153.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_20
    buf156 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_108, buf155, reinterpret_tensor(primals_107, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf156)
    del primals_108
    buf157 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_29(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()))
    buf158 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_4_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf157, reinterpret_tensor(primals_109, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf158)
    del primals_110
    buf159 = buf149; del buf149  # reuse
    buf161 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf163 = reinterpret_tensor(buf161, (4, 128, 1), (128, 1, 1), 0); del buf161  # reuse
    buf164 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf165 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_30(c_void_p(buf163.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del primals_22
    buf166 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf165, reinterpret_tensor(primals_111, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf166)
    del primals_112
    buf167 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf165, reinterpret_tensor(primals_113, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf167)
    del primals_114
    buf168 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf165, reinterpret_tensor(primals_115, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
    del primals_116
    buf169 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf170 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_31(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    buf171 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf169, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf170, (48, 64, 128), (8192, 128, 1), 0), out=buf171)
    buf172 = buf141; del buf141  # reuse
    buf173 = reinterpret_tensor(buf171, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf171  # reuse
    buf174 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf175 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf176 = reinterpret_tensor(buf167, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf167  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_32(c_void_p(buf173.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()))
    buf177 = reinterpret_tensor(buf168, (48, 128, 64), (8192, 64, 1), 0); del buf168  # reuse
    # Source Nodes: [x_45], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf175, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf176, (48, 128, 64), (8192, 64, 1), 0), out=buf177)
    buf178 = buf166; del buf166  # reuse
    cpp_fused_view_33(c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    buf179 = reinterpret_tensor(buf177, (512, 768), (768, 1), 0); del buf177  # reuse
    # Source Nodes: [l__mod___transformer_blocks_5_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_118, buf178, reinterpret_tensor(primals_117, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf179)
    del primals_118
    buf180 = buf159; del buf159  # reuse
    buf182 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf184 = reinterpret_tensor(buf182, (4, 128, 1), (128, 1, 1), 0); del buf182  # reuse
    buf185 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf186 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_34(c_void_p(buf184.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    del primals_24
    buf187 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf186, reinterpret_tensor(primals_119, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf187)
    del primals_120
    buf188 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_35(c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    buf189 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_5_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf188, reinterpret_tensor(primals_121, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf189)
    del primals_122
    buf190 = reinterpret_tensor(buf189, (4, 128, 768), (98304, 768, 1), 0); del buf189  # reuse
    buf191 = buf180; del buf180  # reuse
    buf193 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf195 = reinterpret_tensor(buf193, (4, 128, 1), (128, 1, 1), 0); del buf193  # reuse
    buf196 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf197 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_36(c_void_p(buf190.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    del primals_26
    buf198 = buf179; del buf179  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf197, reinterpret_tensor(primals_123, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf198)
    del primals_124
    buf199 = buf158; del buf158  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_126, buf197, reinterpret_tensor(primals_125, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf199)
    del primals_126
    buf200 = buf148; del buf148  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_128, buf197, reinterpret_tensor(primals_127, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf200)
    del primals_128
    buf201 = reinterpret_tensor(buf127, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf127  # reuse
    buf202 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_37(c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf201, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf202, (48, 64, 128), (8192, 128, 1), 0), out=buf203)
    buf204 = buf172; del buf172  # reuse
    buf205 = reinterpret_tensor(buf203, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf203  # reuse
    buf206 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf207 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf208 = reinterpret_tensor(buf199, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf199  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_38(c_void_p(buf205.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    buf209 = reinterpret_tensor(buf200, (48, 128, 64), (8192, 64, 1), 0); del buf200  # reuse
    # Source Nodes: [x_53], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf207, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf208, (48, 128, 64), (8192, 64, 1), 0), out=buf209)
    buf210 = buf198; del buf198  # reuse
    cpp_fused_view_39(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf209, (512, 768), (768, 1), 0); del buf209  # reuse
    # Source Nodes: [l__mod___transformer_blocks_6_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf210, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf211)
    del primals_130
    buf212 = buf191; del buf191  # reuse
    buf214 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf216 = reinterpret_tensor(buf214, (4, 128, 1), (128, 1, 1), 0); del buf214  # reuse
    buf217 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf218 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_40(c_void_p(buf216.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()))
    del primals_28
    buf219 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf218, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf219)
    del primals_132
    buf220 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_41(c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_6_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf220, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf221)
    del primals_134
    buf222 = buf212; del buf212  # reuse
    buf224 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf224, (4, 128, 1), (128, 1, 1), 0); del buf224  # reuse
    buf227 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf228 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_42(c_void_p(buf226.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    del primals_30
    buf229 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_136, buf228, reinterpret_tensor(primals_135, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf229)
    del primals_136
    buf230 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf228, reinterpret_tensor(primals_137, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf230)
    del primals_138
    buf231 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf228, reinterpret_tensor(primals_139, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf231)
    del primals_140
    buf232 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf233 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_43(c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()))
    buf234 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf232, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf233, (48, 64, 128), (8192, 128, 1), 0), out=buf234)
    buf235 = buf204; del buf204  # reuse
    buf236 = reinterpret_tensor(buf234, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf234  # reuse
    buf237 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf238 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf239 = reinterpret_tensor(buf230, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf230  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_44(c_void_p(buf236.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    buf240 = reinterpret_tensor(buf231, (48, 128, 64), (8192, 64, 1), 0); del buf231  # reuse
    # Source Nodes: [x_61], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf238, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf239, (48, 128, 64), (8192, 64, 1), 0), out=buf240)
    buf241 = buf229; del buf229  # reuse
    cpp_fused_view_45(c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    buf242 = reinterpret_tensor(buf240, (512, 768), (768, 1), 0); del buf240  # reuse
    # Source Nodes: [l__mod___transformer_blocks_7_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf241, reinterpret_tensor(primals_141, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf242)
    del primals_142
    buf243 = buf222; del buf222  # reuse
    buf245 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf247 = reinterpret_tensor(buf245, (4, 128, 1), (128, 1, 1), 0); del buf245  # reuse
    buf248 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf249 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_46(c_void_p(buf247.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del primals_32
    buf250 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf249, reinterpret_tensor(primals_143, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf250)
    del primals_144
    buf251 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_47(c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_7_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_146, buf251, reinterpret_tensor(primals_145, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf252)
    del primals_146
    buf253 = reinterpret_tensor(buf252, (4, 128, 768), (98304, 768, 1), 0); del buf252  # reuse
    buf254 = buf243; del buf243  # reuse
    buf256 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf258 = reinterpret_tensor(buf256, (4, 128, 1), (128, 1, 1), 0); del buf256  # reuse
    buf259 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf260 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_48(c_void_p(buf253.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    del primals_34
    buf261 = buf242; del buf242  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_148, buf260, reinterpret_tensor(primals_147, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf261)
    del primals_148
    buf262 = buf221; del buf221  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf260, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf262)
    del primals_150
    buf263 = buf211; del buf211  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf260, reinterpret_tensor(primals_151, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf263)
    del primals_152
    buf264 = reinterpret_tensor(buf190, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf190  # reuse
    buf265 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_49(c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    buf266 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf264, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf265, (48, 64, 128), (8192, 128, 1), 0), out=buf266)
    buf267 = buf235; del buf235  # reuse
    buf268 = reinterpret_tensor(buf266, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf266  # reuse
    buf269 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf270 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf271 = reinterpret_tensor(buf262, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf262  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_50(c_void_p(buf268.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()))
    buf272 = reinterpret_tensor(buf263, (48, 128, 64), (8192, 64, 1), 0); del buf263  # reuse
    # Source Nodes: [x_69], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf270, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf271, (48, 128, 64), (8192, 64, 1), 0), out=buf272)
    buf273 = buf261; del buf261  # reuse
    cpp_fused_view_51(c_void_p(buf272.data_ptr()), c_void_p(buf273.data_ptr()))
    buf274 = reinterpret_tensor(buf272, (512, 768), (768, 1), 0); del buf272  # reuse
    # Source Nodes: [l__mod___transformer_blocks_8_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, buf273, reinterpret_tensor(primals_153, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf274)
    del primals_154
    buf275 = buf254; del buf254  # reuse
    buf277 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf279 = reinterpret_tensor(buf277, (4, 128, 1), (128, 1, 1), 0); del buf277  # reuse
    buf280 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf281 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_52(c_void_p(buf279.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()))
    del primals_36
    buf282 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_156, buf281, reinterpret_tensor(primals_155, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf282)
    del primals_156
    buf283 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_53(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()))
    buf284 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_8_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_158, buf283, reinterpret_tensor(primals_157, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf284)
    del primals_158
    buf285 = buf275; del buf275  # reuse
    buf287 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf289 = reinterpret_tensor(buf287, (4, 128, 1), (128, 1, 1), 0); del buf287  # reuse
    buf290 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf291 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_54(c_void_p(buf289.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del primals_38
    buf292 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf291, reinterpret_tensor(primals_159, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf292)
    del primals_160
    buf293 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf291, reinterpret_tensor(primals_161, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf293)
    del primals_162
    buf294 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf291, reinterpret_tensor(primals_163, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf294)
    del primals_164
    buf295 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf296 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_55(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()))
    buf297 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf295, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf296, (48, 64, 128), (8192, 128, 1), 0), out=buf297)
    buf298 = buf267; del buf267  # reuse
    buf299 = reinterpret_tensor(buf297, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf297  # reuse
    buf300 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf301 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf302 = reinterpret_tensor(buf293, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf293  # reuse
    cpp_fused__softmax_clone_div_eq_masked_fill_56(c_void_p(buf299.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = reinterpret_tensor(buf294, (48, 128, 64), (8192, 64, 1), 0); del buf294  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf301, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf302, (48, 128, 64), (8192, 64, 1), 0), out=buf303)
    buf304 = buf292; del buf292  # reuse
    cpp_fused_view_57(c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()))
    buf305 = reinterpret_tensor(buf303, (512, 768), (768, 1), 0); del buf303  # reuse
    # Source Nodes: [l__mod___transformer_blocks_9_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf304, reinterpret_tensor(primals_165, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf305)
    del primals_166
    buf306 = buf285; del buf285  # reuse
    buf308 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf310 = reinterpret_tensor(buf308, (4, 128, 1), (128, 1, 1), 0); del buf308  # reuse
    buf311 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf312 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_58(c_void_p(buf310.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del primals_40
    buf313 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_168, buf312, reinterpret_tensor(primals_167, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf313)
    del primals_168
    buf314 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_59(c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()))
    buf315 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_9_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_170, buf314, reinterpret_tensor(primals_169, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf315)
    del primals_170
    buf316 = reinterpret_tensor(buf315, (4, 128, 768), (98304, 768, 1), 0); del buf315  # reuse
    buf317 = buf306; del buf306  # reuse
    buf319 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf321 = reinterpret_tensor(buf319, (4, 128, 1), (128, 1, 1), 0); del buf319  # reuse
    buf322 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf323 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_60(c_void_p(buf316.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del primals_42
    buf324 = buf305; del buf305  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_172, buf323, reinterpret_tensor(primals_171, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf324)
    del primals_172
    buf325 = buf284; del buf284  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_174, buf323, reinterpret_tensor(primals_173, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf325)
    del primals_174
    buf326 = buf274; del buf274  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf323, reinterpret_tensor(primals_175, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf326)
    del primals_176
    buf327 = reinterpret_tensor(buf253, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf253  # reuse
    buf328 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_61(c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()))
    buf329 = empty((48, 128, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf327, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf328, (48, 64, 128), (8192, 128, 1), 0), out=buf329)
    buf330 = buf298; del buf298  # reuse
    buf331 = reinterpret_tensor(buf329, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf329  # reuse
    buf332 = empty_strided((4, 12, 128, 1), (1536, 128, 1, 6144), device='cpu', dtype=torch.float32)
    buf333 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf381 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf334 = reinterpret_tensor(buf325, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf325  # reuse
    cpp_fused__softmax_clone_detach_div_eq_masked_fill_62(c_void_p(buf331.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf334.data_ptr()))
    buf335 = reinterpret_tensor(buf326, (48, 128, 64), (8192, 64, 1), 0); del buf326  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf333, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf334, (48, 128, 64), (8192, 64, 1), 0), out=buf335)
    buf336 = buf324; del buf324  # reuse
    cpp_fused_view_63(c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = reinterpret_tensor(buf335, (512, 768), (768, 1), 0); del buf335  # reuse
    # Source Nodes: [l__mod___transformer_blocks_10_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf336, reinterpret_tensor(primals_177, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf337)
    del primals_178
    buf338 = buf317; del buf317  # reuse
    buf340 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf342 = reinterpret_tensor(buf340, (4, 128, 1), (128, 1, 1), 0); del buf340  # reuse
    buf343 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf344 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_64(c_void_p(buf342.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del primals_44
    buf345 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf344, reinterpret_tensor(primals_179, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf345)
    del primals_180
    buf346 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_65(c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    buf347 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_10_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf346, reinterpret_tensor(primals_181, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf347)
    del primals_182
    buf348 = buf338; del buf338  # reuse
    buf350 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf352 = reinterpret_tensor(buf350, (4, 128, 1), (128, 1, 1), 0); del buf350  # reuse
    buf353 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf354 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_66(c_void_p(buf352.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del primals_46
    buf355 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_0], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_184, buf354, reinterpret_tensor(primals_183, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf355)
    del primals_184
    buf356 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_186, buf354, reinterpret_tensor(primals_185, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf356)
    del primals_186
    buf357 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_linear_layers_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf354, reinterpret_tensor(primals_187, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf357)
    del primals_188
    buf358 = empty((4, 12, 128, 64), device='cpu', dtype=torch.float32)
    buf359 = empty((4, 12, 64, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_67(c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    buf360 = reinterpret_tensor(buf331, (48, 128, 128), (16384, 128, 1), 0); del buf331  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf358, (48, 128, 64), (8192, 64, 1), 0), reinterpret_tensor(buf359, (48, 64, 128), (8192, 128, 1), 0), out=buf360)
    buf361 = buf332; del buf332  # reuse
    buf362 = reinterpret_tensor(buf360, (4, 12, 128, 128), (196608, 16384, 128, 1), 0); del buf360  # reuse
    buf363 = buf330; del buf330  # reuse
    buf364 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf380 = empty((4, 12, 128, 128), device='cpu', dtype=torch.float32)
    buf365 = reinterpret_tensor(buf356, (4, 12, 128, 64), (98304, 8192, 64, 1), 0); del buf356  # reuse
    cpp_fused__softmax_clone_detach_div_eq_masked_fill_68(c_void_p(buf362.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf365.data_ptr()))
    del buf361
    del buf362
    del buf363
    buf366 = reinterpret_tensor(buf357, (48, 128, 64), (8192, 64, 1), 0); del buf357  # reuse
    # Source Nodes: [x_93], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf364, (48, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf365, (48, 128, 64), (8192, 64, 1), 0), out=buf366)
    buf367 = buf355; del buf355  # reuse
    cpp_fused_view_69(c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    buf368 = reinterpret_tensor(buf366, (512, 768), (768, 1), 0); del buf366  # reuse
    # Source Nodes: [l__mod___transformer_blocks_11_lambda_module_attention_output_linear], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_190, buf367, reinterpret_tensor(primals_189, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf368)
    del primals_190
    buf369 = buf348; del buf348  # reuse
    buf371 = empty_strided((4, 128, 1), (128, 1, 512), device='cpu', dtype=torch.float32)
    buf373 = reinterpret_tensor(buf371, (4, 128, 1), (128, 1, 1), 0); del buf371  # reuse
    buf374 = empty((4, 128, 768), device='cpu', dtype=torch.float32)
    buf375 = empty((512, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_div_mean_mul_std_sub_view_70(c_void_p(buf373.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del buf369
    del primals_48
    buf376 = empty((512, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_w_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_192, buf375, reinterpret_tensor(primals_191, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf376)
    del primals_192
    buf377 = empty((512, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_71(c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()))
    buf378 = empty((512, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___transformer_blocks_11_feed_forward_w_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_194, buf377, reinterpret_tensor(primals_193, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf378)
    del primals_194
    buf379 = reinterpret_tensor(buf378, (4, 128, 768), (98304, 768, 1), 0); del buf378  # reuse
    buf382 = buf299; del buf299  # reuse
    buf383 = buf268; del buf268  # reuse
    buf384 = buf236; del buf236  # reuse
    buf385 = buf205; del buf205  # reuse
    buf386 = buf173; del buf173  # reuse
    buf387 = buf142; del buf142  # reuse
    buf388 = buf110; del buf110  # reuse
    buf389 = buf79; del buf79  # reuse
    buf390 = buf47; del buf47  # reuse
    buf391 = buf16; del buf16  # reuse
    cpp_fused__softmax_add_detach_72(c_void_p(buf379.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(buf17.data_ptr()))
    return (buf379, buf0, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_196, primals_197, buf0, buf6, buf7, buf8, buf21, buf27, buf28, buf29, buf30, buf31, buf37, buf38, buf39, buf52, buf58, buf59, buf60, buf61, buf62, buf69, buf70, buf71, buf84, buf90, buf91, buf92, buf93, buf94, buf100, buf101, buf102, buf115, buf121, buf122, buf123, buf124, buf125, buf132, buf133, buf134, buf147, buf153, buf154, buf155, buf156, buf157, buf163, buf164, buf165, buf178, buf184, buf185, buf186, buf187, buf188, buf195, buf196, buf197, buf210, buf216, buf217, buf218, buf219, buf220, buf226, buf227, buf228, buf241, buf247, buf248, buf249, buf250, buf251, buf258, buf259, buf260, buf273, buf279, buf280, buf281, buf282, buf283, buf289, buf290, buf291, buf304, buf310, buf311, buf312, buf313, buf314, buf321, buf322, buf323, buf336, buf342, buf343, buf344, buf345, buf346, buf352, buf353, buf354, buf367, buf373, buf374, buf375, buf376, buf377, reinterpret_tensor(primals_193, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_191, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_189, (768, 768), (768, 1), 0), reinterpret_tensor(buf364, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf365, (48, 64, 128), (8192, 1, 64), 0), buf380, reinterpret_tensor(buf358, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf359, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_187, (768, 768), (768, 1), 0), reinterpret_tensor(primals_185, (768, 768), (768, 1), 0), reinterpret_tensor(primals_183, (768, 768), (768, 1), 0), reinterpret_tensor(primals_181, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_179, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 768), (768, 1), 0), reinterpret_tensor(buf333, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf334, (48, 64, 128), (8192, 1, 64), 0), buf381, reinterpret_tensor(buf327, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf328, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_175, (768, 768), (768, 1), 0), reinterpret_tensor(primals_173, (768, 768), (768, 1), 0), reinterpret_tensor(primals_171, (768, 768), (768, 1), 0), reinterpret_tensor(primals_169, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_167, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_165, (768, 768), (768, 1), 0), reinterpret_tensor(buf301, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf302, (48, 64, 128), (8192, 1, 64), 0), buf382, reinterpret_tensor(buf295, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf296, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_163, (768, 768), (768, 1), 0), reinterpret_tensor(primals_161, (768, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 768), (768, 1), 0), reinterpret_tensor(primals_157, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_155, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_153, (768, 768), (768, 1), 0), reinterpret_tensor(buf270, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf271, (48, 64, 128), (8192, 1, 64), 0), buf383, reinterpret_tensor(buf264, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf265, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_151, (768, 768), (768, 1), 0), reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 768), (768, 1), 0), reinterpret_tensor(primals_145, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_143, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_141, (768, 768), (768, 1), 0), reinterpret_tensor(buf238, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf239, (48, 64, 128), (8192, 1, 64), 0), buf384, reinterpret_tensor(buf232, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf233, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_139, (768, 768), (768, 1), 0), reinterpret_tensor(primals_137, (768, 768), (768, 1), 0), reinterpret_tensor(primals_135, (768, 768), (768, 1), 0), reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), reinterpret_tensor(buf207, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf208, (48, 64, 128), (8192, 1, 64), 0), buf385, reinterpret_tensor(buf201, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf202, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_127, (768, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 768), (768, 1), 0), reinterpret_tensor(primals_123, (768, 768), (768, 1), 0), reinterpret_tensor(primals_121, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_119, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_117, (768, 768), (768, 1), 0), reinterpret_tensor(buf175, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf176, (48, 64, 128), (8192, 1, 64), 0), buf386, reinterpret_tensor(buf169, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf170, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_115, (768, 768), (768, 1), 0), reinterpret_tensor(primals_113, (768, 768), (768, 1), 0), reinterpret_tensor(primals_111, (768, 768), (768, 1), 0), reinterpret_tensor(primals_109, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_107, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_105, (768, 768), (768, 1), 0), reinterpret_tensor(buf144, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf145, (48, 64, 128), (8192, 1, 64), 0), buf387, reinterpret_tensor(buf138, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf139, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_103, (768, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 768), (768, 1), 0), reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_95, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_93, (768, 768), (768, 1), 0), reinterpret_tensor(buf112, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf113, (48, 64, 128), (8192, 1, 64), 0), buf388, reinterpret_tensor(buf106, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf107, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_91, (768, 768), (768, 1), 0), reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(primals_87, (768, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_83, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_81, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf82, (48, 64, 128), (8192, 1, 64), 0), buf389, reinterpret_tensor(buf75, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf76, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_79, (768, 768), (768, 1), 0), reinterpret_tensor(primals_77, (768, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (768, 1), 0), reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(buf49, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf50, (48, 64, 128), (8192, 1, 64), 0), buf390, reinterpret_tensor(buf43, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf44, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_67, (768, 768), (768, 1), 0), reinterpret_tensor(primals_65, (768, 768), (768, 1), 0), reinterpret_tensor(primals_63, (768, 768), (768, 1), 0), reinterpret_tensor(primals_61, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 768), (768, 1), 0), reinterpret_tensor(buf18, (48, 128, 128), (16384, 1, 128), 0), reinterpret_tensor(buf19, (48, 64, 128), (8192, 1, 64), 0), buf391, reinterpret_tensor(buf12, (48, 64, 128), (8192, 1, 64), 0), reinterpret_tensor(buf13, (48, 128, 64), (8192, 1, 128), 0), reinterpret_tensor(primals_55, (768, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 768), (768, 1), 0), reinterpret_tensor(primals_51, (768, 768), (768, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((20005, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((3, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((1, 512, 768), (393216, 768, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    primals_197 = rand_strided((4, 128), (128, 1), device='cpu', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('BERT_pytorch', benchmark_compiled_module)
