
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


cpp_fused_add_copy_native_layer_norm_select_scatter_view_zeros_1 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(768.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-06);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        tmp13.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 == tmp1;
                        auto tmp3 = c10::convert<long>(((-1L)*(static_cast<long>(x0) % static_cast<long>(14L))) + (static_cast<long>(x1) % static_cast<long>(14L)));
                        auto tmp4 = c10::convert<float>(tmp3);
                        auto tmp5 = static_cast<int>(1);
                        auto tmp6 = tmp0 == tmp5;
                        auto tmp7 = c10::convert<long>(((-1L)*(c10::div_floor_integer(x0, 14L))) + (c10::div_floor_integer(x1, 14L)));
                        auto tmp8 = c10::convert<float>(tmp7);
                        auto tmp9 = static_cast<int>(2);
                        auto tmp10 = tmp0 == tmp9;
                        auto tmp11 = c10::convert<long>((static_cast<long>((c10::div_floor_integer(x0, 14L))*(c10::div_floor_integer(x0, 14L)))) + (static_cast<long>((c10::div_floor_integer(x1, 14L))*(c10::div_floor_integer(x1, 14L)))) + (static_cast<long>((static_cast<long>(x0) % static_cast<long>(14L))*(static_cast<long>(x0) % static_cast<long>(14L)))) + (static_cast<long>((static_cast<long>(x1) % static_cast<long>(14L))*(static_cast<long>(x1) % static_cast<long>(14L)))) + ((-2L)*(c10::div_floor_integer(x0, 14L))*(c10::div_floor_integer(x1, 14L))) + ((-2L)*(static_cast<long>(x0) % static_cast<long>(14L))*(static_cast<long>(x1) % static_cast<long>(14L))));
                        auto tmp12 = c10::convert<float>(tmp11);
                        auto tmp13 = static_cast<float>(0.0);
                        auto tmp14 = tmp10 ? tmp12 : tmp13;
                        auto tmp15 = tmp6 ? tmp8 : tmp14;
                        auto tmp16 = tmp2 ? tmp4 : tmp15;
                        out_ptr3[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
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
                    tmp4.store(out_ptr4 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(307328L); x0+=static_cast<long>(1L))
            {
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (3L*(static_cast<long>(x0) % static_cast<long>(38416L))))];
                    out_ptr0[static_cast<long>(x1 + (3L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_clone_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_6 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp4);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
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
                        tmp15.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
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
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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


cpp_fused_add_native_layer_norm_view_9 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            auto tmp6 = tmp4 + tmp5;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp6);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.mean);
                        out_ptr1[static_cast<long>(x1 + (196L*x0))] = static_cast<float>(tmp_acc0.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(768.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-06);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
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


cpp_fused_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_13 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(150528L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (150528L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (150528L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (150528L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (150528L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (150528L*x0)));
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_15 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_16 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_20 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_22 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_view_30 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_34 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_view_37 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_41 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_43 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_44 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_48 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_49 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_50 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_51 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_55 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_57 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_58 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_62 = async_compile.cpp('''
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_64 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_view_65 = async_compile.cpp('''
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (1536L*x2) + (301056L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)), static_cast<long>(1536L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (1536L*x2) + (301056L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (150528L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto in_ptr0 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr1 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                            float tmp_acc0 = 0;
                            at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                                auto tmp2 = tmp0 / tmp1;
                                auto tmp5 = tmp3 / tmp4;
                                auto tmp7 = decltype(tmp6)(1)/(decltype(tmp6)(1) + tmp6.neg().exp());
                                auto tmp8 = static_cast<float>(1.0);
                                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                                auto tmp10 = tmp9 - tmp7;
                                auto tmp11 = tmp10 * tmp2;
                                auto tmp12 = tmp7 * tmp5;
                                auto tmp13 = tmp11 + tmp12;
                                tmp2.store(out_ptr4 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr5[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                                tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            }
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp_acc0_vec.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr6[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (3136L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr7 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_div_mul_rsub_sigmoid_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)), static_cast<long>(16L), tmp4, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = at::vec::Vectorized<float>::loadu(tmp4 + static_cast<long>(8L*x1_inner));
                                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                                auto tmp12 = in_ptr3[static_cast<long>(x1 + x1_inner + (16L*x2) + (3136L*x0))];
                                auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                                auto tmp2 = static_cast<float>(1.0);
                                auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                                auto tmp6 = at::vec::Vectorized<float>(tmp3);
                                auto tmp7 = tmp6 * tmp5;
                                auto tmp9 = at::vec::Vectorized<float>(tmp1);
                                auto tmp10 = tmp9 * tmp8;
                                auto tmp11 = tmp7 + tmp10;
                                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                                auto tmp14 = tmp11 / tmp13;
                                tmp14.store(out_ptr0 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0)));
                            auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp3 - tmp1;
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = tmp1 * tmp7;
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp11 = tmp9 / tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (38416L*x1_inner) + (614656L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr1 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(196L))) + (9408L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (150528L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
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
                    tmp11.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_cat_native_layer_norm_view_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        tmp17.store(out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.m2);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(768.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-06);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp4 = in_out_ptr0[static_cast<long>(x0)];
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp8 = tmp6 * tmp7;
                    auto tmp10 = tmp8 + tmp9;
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (48L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(197L))) + (9456L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (151296L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(768.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-06);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp13.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp17.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_78 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp19.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)), static_cast<long>(2304L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(768L + x1 + (2304L*x2) + (453888L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (151296L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 - tmp5;
                        auto tmp7 = tmp6.exp();
                        tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.14433756729740643);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                        auto tmp5 = std::exp(tmp4);
                        in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp5;
                        tmp_acc0 = tmp_acc0 + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25216L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(1536L + x3 + (48L*x1) + (2304L*x2) + (453888L*x0)));
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((48L*(static_cast<long>(x0) % static_cast<long>(197L))) + (9456L*(c10::div_floor_integer((x1 + x1_inner), 48L))) + (151296L*(c10::div_floor_integer(x0, 197L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(48L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(768.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-06);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    tmp21.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_clone_native_layer_norm_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1210368L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (151296L*x0)));
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
}
''')


cpp_fused__softmax_add_detach_native_layer_norm_native_layer_norm_backward_85 = async_compile.cpp('''
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
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr0[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
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
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1576L); x0+=static_cast<long>(8L))
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
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0)));
                            auto tmp1 = in_ptr3[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr1[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3152L*x2) + (620944L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(197L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x3 + (197L*x2) + (38809L*x1) + (620944L*x0))];
                            auto tmp1 = in_ptr3[static_cast<long>(x2 + (197L*x1) + (3152L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr1[static_cast<long>(x1 + (16L*x3) + (3152L*x2) + (620944L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
        #pragma omp single
        {
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
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181 = args
    args.clear()
    assert_size_stride(primals_1, (1, 196, 768), (150528, 768, 1))
    assert_size_stride(primals_2, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (768, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (768, ), (1, ))
    assert_size_stride(primals_9, (768, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (768, ), (1, ))
    assert_size_stride(primals_14, (768, ), (1, ))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (768, ), (1, ))
    assert_size_stride(primals_25, (16, ), (1, ))
    assert_size_stride(primals_26, (768, ), (1, ))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (16, ), (1, ))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (16, ), (1, ))
    assert_size_stride(primals_36, (768, ), (1, ))
    assert_size_stride(primals_37, (768, ), (1, ))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (16, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (16, ), (1, ))
    assert_size_stride(primals_46, (768, ), (1, ))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_48, (768, ), (1, ))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (16, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (768, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (768, ), (1, ))
    assert_size_stride(primals_58, (768, ), (1, ))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_64, (768, ), (1, ))
    assert_size_stride(primals_65, (1536, 768), (768, 1))
    assert_size_stride(primals_66, (16, 3), (3, 1))
    assert_size_stride(primals_67, (16, ), (1, ))
    assert_size_stride(primals_68, (768, 768), (768, 1))
    assert_size_stride(primals_69, (768, 768), (768, 1))
    assert_size_stride(primals_70, (768, ), (1, ))
    assert_size_stride(primals_71, (3072, 768), (768, 1))
    assert_size_stride(primals_72, (3072, ), (1, ))
    assert_size_stride(primals_73, (768, 3072), (3072, 1))
    assert_size_stride(primals_74, (768, ), (1, ))
    assert_size_stride(primals_75, (1536, 768), (768, 1))
    assert_size_stride(primals_76, (16, 3), (3, 1))
    assert_size_stride(primals_77, (16, ), (1, ))
    assert_size_stride(primals_78, (768, 768), (768, 1))
    assert_size_stride(primals_79, (768, 768), (768, 1))
    assert_size_stride(primals_80, (768, ), (1, ))
    assert_size_stride(primals_81, (3072, 768), (768, 1))
    assert_size_stride(primals_82, (3072, ), (1, ))
    assert_size_stride(primals_83, (768, 3072), (3072, 1))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (1536, 768), (768, 1))
    assert_size_stride(primals_86, (16, 3), (3, 1))
    assert_size_stride(primals_87, (16, ), (1, ))
    assert_size_stride(primals_88, (768, 768), (768, 1))
    assert_size_stride(primals_89, (768, 768), (768, 1))
    assert_size_stride(primals_90, (768, ), (1, ))
    assert_size_stride(primals_91, (3072, 768), (768, 1))
    assert_size_stride(primals_92, (3072, ), (1, ))
    assert_size_stride(primals_93, (768, 3072), (3072, 1))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (1536, 768), (768, 1))
    assert_size_stride(primals_96, (16, 3), (3, 1))
    assert_size_stride(primals_97, (16, ), (1, ))
    assert_size_stride(primals_98, (768, 768), (768, 1))
    assert_size_stride(primals_99, (768, 768), (768, 1))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (1536, 768), (768, 1))
    assert_size_stride(primals_106, (16, 3), (3, 1))
    assert_size_stride(primals_107, (16, ), (1, ))
    assert_size_stride(primals_108, (768, 768), (768, 1))
    assert_size_stride(primals_109, (768, 768), (768, 1))
    assert_size_stride(primals_110, (768, ), (1, ))
    assert_size_stride(primals_111, (3072, 768), (768, 1))
    assert_size_stride(primals_112, (3072, ), (1, ))
    assert_size_stride(primals_113, (768, 3072), (3072, 1))
    assert_size_stride(primals_114, (768, ), (1, ))
    assert_size_stride(primals_115, (1536, 768), (768, 1))
    assert_size_stride(primals_116, (16, 3), (3, 1))
    assert_size_stride(primals_117, (16, ), (1, ))
    assert_size_stride(primals_118, (768, 768), (768, 1))
    assert_size_stride(primals_119, (768, 768), (768, 1))
    assert_size_stride(primals_120, (768, ), (1, ))
    assert_size_stride(primals_121, (3072, 768), (768, 1))
    assert_size_stride(primals_122, (3072, ), (1, ))
    assert_size_stride(primals_123, (768, 3072), (3072, 1))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (1536, 768), (768, 1))
    assert_size_stride(primals_126, (16, 3), (3, 1))
    assert_size_stride(primals_127, (16, ), (1, ))
    assert_size_stride(primals_128, (768, 768), (768, 1))
    assert_size_stride(primals_129, (768, 768), (768, 1))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (3072, 768), (768, 1))
    assert_size_stride(primals_132, (3072, ), (1, ))
    assert_size_stride(primals_133, (768, 3072), (3072, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (1536, 768), (768, 1))
    assert_size_stride(primals_136, (16, 3), (3, 1))
    assert_size_stride(primals_137, (16, ), (1, ))
    assert_size_stride(primals_138, (768, 768), (768, 1))
    assert_size_stride(primals_139, (768, 768), (768, 1))
    assert_size_stride(primals_140, (768, ), (1, ))
    assert_size_stride(primals_141, (3072, 768), (768, 1))
    assert_size_stride(primals_142, (3072, ), (1, ))
    assert_size_stride(primals_143, (768, 3072), (3072, 1))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (1536, 768), (768, 1))
    assert_size_stride(primals_146, (16, 3), (3, 1))
    assert_size_stride(primals_147, (16, ), (1, ))
    assert_size_stride(primals_148, (768, 768), (768, 1))
    assert_size_stride(primals_149, (768, 768), (768, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (3072, 768), (768, 1))
    assert_size_stride(primals_152, (3072, ), (1, ))
    assert_size_stride(primals_153, (768, 3072), (3072, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (1536, 768), (768, 1))
    assert_size_stride(primals_156, (16, 3), (3, 1))
    assert_size_stride(primals_157, (16, ), (1, ))
    assert_size_stride(primals_158, (768, 768), (768, 1))
    assert_size_stride(primals_159, (768, 768), (768, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (3072, 768), (768, 1))
    assert_size_stride(primals_162, (3072, ), (1, ))
    assert_size_stride(primals_163, (768, 3072), (3072, 1))
    assert_size_stride(primals_164, (768, ), (1, ))
    assert_size_stride(primals_165, (2304, 768), (768, 1))
    assert_size_stride(primals_166, (768, 768), (768, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (3072, 768), (768, 1))
    assert_size_stride(primals_169, (3072, ), (1, ))
    assert_size_stride(primals_170, (768, 3072), (3072, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_172, (2304, 768), (768, 1))
    assert_size_stride(primals_173, (768, 768), (768, 1))
    assert_size_stride(primals_174, (768, ), (1, ))
    assert_size_stride(primals_175, (3072, 768), (768, 1))
    assert_size_stride(primals_176, (3072, ), (1, ))
    assert_size_stride(primals_177, (768, 3072), (3072, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (1000, 768), (768, 1))
    assert_size_stride(primals_180, (1000, ), (1, ))
    assert_size_stride(primals_181, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_63.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del primals_181
    del primals_63
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf1, buf0, primals_64, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del primals_64
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf7 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf8 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_copy_native_layer_norm_select_scatter_view_zeros_1(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del primals_4
    buf9 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf8, reinterpret_tensor(primals_65, (768, 1536), (1, 768), 0), out=buf9)
    buf10 = empty((307328, 3), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_2(c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    buf11 = empty((307328, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_66, (3, 16), (1, 3), 0), out=buf11)
    del primals_66
    buf12 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf13 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_3(c_void_p(buf9.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    buf14 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf12, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf13, (128, 48, 196), (9408, 196, 1), 0), out=buf14)
    buf15 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf16 = reinterpret_tensor(buf14, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf14  # reuse
    buf17 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf19 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 25088), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf11, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf11  # reuse
    buf21 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf22 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf23 = empty((8, 16, 196), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_4(c_void_p(buf16.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    del primals_67
    buf25 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf8, reinterpret_tensor(primals_68, (768, 768), (1, 768), 0), out=buf25)
    buf26 = reinterpret_tensor(buf20, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf20  # reuse
    buf27 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_5(c_void_p(primals_5.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    buf28 = reinterpret_tensor(buf25, (128, 196, 48), (9408, 48, 1), 0); del buf25  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf26, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf27, (128, 196, 48), (9408, 48, 1), 0), out=buf28)
    buf29 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_6(c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = reinterpret_tensor(buf28, (1568, 768), (768, 1), 0); del buf28  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_70, buf29, reinterpret_tensor(primals_69, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf30)
    del primals_70
    buf31 = buf3; del buf3  # reuse
    buf32 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf34 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf35 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_7(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_7
    buf36 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf35, reinterpret_tensor(primals_71, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf36)
    del primals_72
    buf37 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_8(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    buf38 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_74, buf37, reinterpret_tensor(primals_73, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf38)
    del primals_74
    buf39 = buf31; del buf31  # reuse
    buf40 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf42 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf43 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_9(c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del primals_9
    buf44 = buf9; del buf9  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, reinterpret_tensor(primals_75, (768, 1536), (1, 768), 0), out=buf44)
    buf45 = reinterpret_tensor(buf16, (307328, 16), (16, 1), 0); del buf16  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_76, (3, 16), (1, 3), 0), out=buf45)
    del primals_76
    buf46 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf47 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_10(c_void_p(buf44.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf46, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf47, (128, 48, 196), (9408, 196, 1), 0), out=buf48)
    buf49 = reinterpret_tensor(buf23, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf23  # reuse
    buf50 = reinterpret_tensor(buf48, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf48  # reuse
    buf51 = buf21; del buf21  # reuse
    buf53 = buf19; del buf19  # reuse
    buf54 = reinterpret_tensor(buf45, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf45  # reuse
    buf55 = buf17; del buf17  # reuse
    buf52 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf56 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf57 = reinterpret_tensor(buf15, (8, 16, 196), (3136, 196, 1), 0); del buf15  # reuse
    buf58 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_11(c_void_p(buf50.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    del primals_77
    buf59 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_1_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf43, reinterpret_tensor(primals_78, (768, 768), (1, 768), 0), out=buf59)
    buf60 = reinterpret_tensor(buf54, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf54  # reuse
    buf61 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_12(c_void_p(primals_10.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf59, (128, 196, 48), (9408, 48, 1), 0); del buf59  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf60, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf61, (128, 196, 48), (9408, 48, 1), 0), out=buf62)
    buf63 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_13(c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = reinterpret_tensor(buf62, (1568, 768), (768, 1), 0); del buf62  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_80, buf63, reinterpret_tensor(primals_79, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf64)
    del primals_80
    buf65 = reinterpret_tensor(buf64, (8, 196, 768), (150528, 768, 1), 0); del buf64  # reuse
    buf66 = buf39; del buf39  # reuse
    buf67 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf69 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf70 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_14(c_void_p(buf65.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del primals_1
    del primals_12
    buf71 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_82, buf70, reinterpret_tensor(primals_81, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf71)
    del primals_82
    buf72 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_15(c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    buf73 = buf38; del buf38  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_84, buf72, reinterpret_tensor(primals_83, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf73)
    del primals_84
    buf74 = buf66; del buf66  # reuse
    buf75 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf77 = reinterpret_tensor(buf30, (8, 196, 768), (150528, 768, 1), 0); del buf30  # reuse
    buf78 = reinterpret_tensor(buf2, (1568, 768), (768, 1), 0); del buf2  # reuse
    cpp_fused_add_native_layer_norm_view_16(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del primals_14
    buf79 = buf44; del buf44  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf78, reinterpret_tensor(primals_85, (768, 1536), (1, 768), 0), out=buf79)
    buf80 = reinterpret_tensor(buf50, (307328, 16), (16, 1), 0); del buf50  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_86, (3, 16), (1, 3), 0), out=buf80)
    del primals_86
    buf81 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf82 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_17(c_void_p(buf79.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf82, (128, 48, 196), (9408, 196, 1), 0), out=buf83)
    buf84 = reinterpret_tensor(buf57, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf57  # reuse
    buf85 = reinterpret_tensor(buf83, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf83  # reuse
    buf86 = buf55; del buf55  # reuse
    buf88 = buf53; del buf53  # reuse
    buf89 = reinterpret_tensor(buf80, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf80  # reuse
    buf90 = buf51; del buf51  # reuse
    buf87 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf91 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf92 = reinterpret_tensor(buf49, (8, 16, 196), (3136, 196, 1), 0); del buf49  # reuse
    buf93 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_18(c_void_p(buf85.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()))
    del primals_87
    buf94 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_2_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf78, reinterpret_tensor(primals_88, (768, 768), (1, 768), 0), out=buf94)
    buf95 = reinterpret_tensor(buf89, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf89  # reuse
    buf96 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_19(c_void_p(primals_15.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()))
    buf97 = reinterpret_tensor(buf94, (128, 196, 48), (9408, 48, 1), 0); del buf94  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf95, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf96, (128, 196, 48), (9408, 48, 1), 0), out=buf97)
    buf98 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_20(c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()))
    buf99 = reinterpret_tensor(buf97, (1568, 768), (768, 1), 0); del buf97  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_90, buf98, reinterpret_tensor(primals_89, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf99)
    del primals_90
    buf100 = buf74; del buf74  # reuse
    buf101 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf103 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf104 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_21(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()))
    del primals_17
    buf105 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_92, buf104, reinterpret_tensor(primals_91, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf105)
    del primals_92
    buf106 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_22(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf106, reinterpret_tensor(primals_93, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf107)
    del primals_94
    buf108 = buf100; del buf100  # reuse
    buf109 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf111 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf112 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    del primals_19
    buf113 = buf79; del buf79  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf112, reinterpret_tensor(primals_95, (768, 1536), (1, 768), 0), out=buf113)
    buf114 = reinterpret_tensor(buf85, (307328, 16), (16, 1), 0); del buf85  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_96, (3, 16), (1, 3), 0), out=buf114)
    del primals_96
    buf115 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf116 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_24(c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf116, (128, 48, 196), (9408, 196, 1), 0), out=buf117)
    buf118 = reinterpret_tensor(buf92, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf92  # reuse
    buf119 = reinterpret_tensor(buf117, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf117  # reuse
    buf120 = buf90; del buf90  # reuse
    buf122 = buf88; del buf88  # reuse
    buf123 = reinterpret_tensor(buf114, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf114  # reuse
    buf124 = buf86; del buf86  # reuse
    buf121 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf125 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf126 = reinterpret_tensor(buf84, (8, 16, 196), (3136, 196, 1), 0); del buf84  # reuse
    buf127 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_25(c_void_p(buf119.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del primals_97
    buf128 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_3_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf112, reinterpret_tensor(primals_98, (768, 768), (1, 768), 0), out=buf128)
    buf129 = reinterpret_tensor(buf123, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf123  # reuse
    buf130 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_26(c_void_p(primals_20.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    buf131 = reinterpret_tensor(buf128, (128, 196, 48), (9408, 48, 1), 0); del buf128  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf129, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf130, (128, 196, 48), (9408, 48, 1), 0), out=buf131)
    buf132 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_27(c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf131, (1568, 768), (768, 1), 0); del buf131  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf132, reinterpret_tensor(primals_99, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf133)
    del primals_100
    buf134 = reinterpret_tensor(buf133, (8, 196, 768), (150528, 768, 1), 0); del buf133  # reuse
    buf135 = buf108; del buf108  # reuse
    buf136 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf138 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf139 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_28(c_void_p(buf134.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    del primals_22
    buf140 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_102, buf139, reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf140)
    del primals_102
    buf141 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_29(c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()))
    buf142 = buf99; del buf99  # reuse
    # Source Nodes: [x_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_104, buf141, reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf142)
    del primals_104
    buf143 = buf135; del buf135  # reuse
    buf144 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf73, (8, 196, 768), (150528, 768, 1), 0); del buf73  # reuse
    buf147 = reinterpret_tensor(buf65, (1568, 768), (768, 1), 0); del buf65  # reuse
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    del primals_24
    buf148 = buf113; del buf113  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf147, reinterpret_tensor(primals_105, (768, 1536), (1, 768), 0), out=buf148)
    buf149 = reinterpret_tensor(buf119, (307328, 16), (16, 1), 0); del buf119  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_106, (3, 16), (1, 3), 0), out=buf149)
    del primals_106
    buf150 = reinterpret_tensor(buf107, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf107  # reuse
    buf151 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_31(c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf150, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf151, (128, 48, 196), (9408, 196, 1), 0), out=buf152)
    buf153 = reinterpret_tensor(buf126, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf126  # reuse
    buf154 = reinterpret_tensor(buf152, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf152  # reuse
    buf155 = buf124; del buf124  # reuse
    buf157 = buf122; del buf122  # reuse
    buf158 = reinterpret_tensor(buf149, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf149  # reuse
    buf159 = buf120; del buf120  # reuse
    buf156 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf160 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf161 = reinterpret_tensor(buf118, (8, 16, 196), (3136, 196, 1), 0); del buf118  # reuse
    buf162 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_32(c_void_p(buf154.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    del primals_107
    buf163 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_4_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf147, reinterpret_tensor(primals_108, (768, 768), (1, 768), 0), out=buf163)
    buf164 = reinterpret_tensor(buf158, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf158  # reuse
    buf165 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_33(c_void_p(primals_25.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    buf166 = reinterpret_tensor(buf163, (128, 196, 48), (9408, 48, 1), 0); del buf163  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf164, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf165, (128, 196, 48), (9408, 48, 1), 0), out=buf166)
    buf167 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_34(c_void_p(buf166.data_ptr()), c_void_p(buf167.data_ptr()))
    buf168 = reinterpret_tensor(buf166, (1568, 768), (768, 1), 0); del buf166  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_110, buf167, reinterpret_tensor(primals_109, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf168)
    del primals_110
    buf169 = buf143; del buf143  # reuse
    buf170 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf172 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf173 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_35(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_27
    buf174 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_112, buf173, reinterpret_tensor(primals_111, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf174)
    del primals_112
    buf175 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_36(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()))
    buf176 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_114, buf175, reinterpret_tensor(primals_113, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf176)
    del primals_114
    buf177 = buf169; del buf169  # reuse
    buf178 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf180 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf181 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del primals_29
    buf182 = buf148; del buf148  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf181, reinterpret_tensor(primals_115, (768, 1536), (1, 768), 0), out=buf182)
    buf183 = reinterpret_tensor(buf154, (307328, 16), (16, 1), 0); del buf154  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_116, (3, 16), (1, 3), 0), out=buf183)
    del primals_116
    buf184 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf185 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_38(c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()))
    buf186 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf184, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf185, (128, 48, 196), (9408, 196, 1), 0), out=buf186)
    buf187 = reinterpret_tensor(buf161, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf161  # reuse
    buf188 = reinterpret_tensor(buf186, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf186  # reuse
    buf189 = buf159; del buf159  # reuse
    buf191 = buf157; del buf157  # reuse
    buf192 = reinterpret_tensor(buf183, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf183  # reuse
    buf193 = buf155; del buf155  # reuse
    buf190 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf194 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf195 = reinterpret_tensor(buf153, (8, 16, 196), (3136, 196, 1), 0); del buf153  # reuse
    buf196 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_39(c_void_p(buf188.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()))
    del primals_117
    buf197 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_5_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf181, reinterpret_tensor(primals_118, (768, 768), (1, 768), 0), out=buf197)
    buf198 = reinterpret_tensor(buf192, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf192  # reuse
    buf199 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_40(c_void_p(primals_30.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = reinterpret_tensor(buf197, (128, 196, 48), (9408, 48, 1), 0); del buf197  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf198, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf199, (128, 196, 48), (9408, 48, 1), 0), out=buf200)
    buf201 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_41(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()))
    buf202 = reinterpret_tensor(buf200, (1568, 768), (768, 1), 0); del buf200  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf201, reinterpret_tensor(primals_119, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf202)
    del primals_120
    buf203 = reinterpret_tensor(buf202, (8, 196, 768), (150528, 768, 1), 0); del buf202  # reuse
    buf204 = buf177; del buf177  # reuse
    buf205 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf207 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf208 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_42(c_void_p(buf203.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del primals_32
    buf209 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf208, reinterpret_tensor(primals_121, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf209)
    del primals_122
    buf210 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_43(c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = buf176; del buf176  # reuse
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_124, buf210, reinterpret_tensor(primals_123, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf211)
    del primals_124
    buf212 = buf204; del buf204  # reuse
    buf213 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf215 = reinterpret_tensor(buf168, (8, 196, 768), (150528, 768, 1), 0); del buf168  # reuse
    buf216 = buf142; del buf142  # reuse
    cpp_fused_add_native_layer_norm_view_44(c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del primals_34
    buf217 = buf182; del buf182  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf216, reinterpret_tensor(primals_125, (768, 1536), (1, 768), 0), out=buf217)
    buf218 = reinterpret_tensor(buf188, (307328, 16), (16, 1), 0); del buf188  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_126, (3, 16), (1, 3), 0), out=buf218)
    del primals_126
    buf219 = reinterpret_tensor(buf134, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf134  # reuse
    buf220 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_45(c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()))
    buf221 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf220, (128, 48, 196), (9408, 196, 1), 0), out=buf221)
    buf222 = reinterpret_tensor(buf195, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf195  # reuse
    buf223 = reinterpret_tensor(buf221, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf221  # reuse
    buf224 = buf193; del buf193  # reuse
    buf226 = buf191; del buf191  # reuse
    buf227 = reinterpret_tensor(buf218, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf218  # reuse
    buf228 = buf189; del buf189  # reuse
    buf225 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf229 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf230 = reinterpret_tensor(buf187, (8, 16, 196), (3136, 196, 1), 0); del buf187  # reuse
    buf231 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_46(c_void_p(buf223.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_127
    buf232 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_6_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf216, reinterpret_tensor(primals_128, (768, 768), (1, 768), 0), out=buf232)
    buf233 = reinterpret_tensor(buf227, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf227  # reuse
    buf234 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_47(c_void_p(primals_35.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = reinterpret_tensor(buf232, (128, 196, 48), (9408, 48, 1), 0); del buf232  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf234, (128, 196, 48), (9408, 48, 1), 0), out=buf235)
    buf236 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_48(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf235, (1568, 768), (768, 1), 0); del buf235  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_130, buf236, reinterpret_tensor(primals_129, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf237)
    del primals_130
    buf238 = buf212; del buf212  # reuse
    buf239 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf241 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf242 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_49(c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del primals_37
    buf243 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_132, buf242, reinterpret_tensor(primals_131, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf243)
    del primals_132
    buf244 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_50(c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()))
    buf245 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_134, buf244, reinterpret_tensor(primals_133, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf245)
    del primals_134
    buf246 = buf238; del buf238  # reuse
    buf247 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf249 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf250 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del primals_39
    buf251 = buf217; del buf217  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, reinterpret_tensor(primals_135, (768, 1536), (1, 768), 0), out=buf251)
    buf252 = reinterpret_tensor(buf223, (307328, 16), (16, 1), 0); del buf223  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_136, (3, 16), (1, 3), 0), out=buf252)
    del primals_136
    buf253 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf254 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_52(c_void_p(buf251.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    buf255 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf253, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf254, (128, 48, 196), (9408, 196, 1), 0), out=buf255)
    buf256 = reinterpret_tensor(buf230, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf230  # reuse
    buf257 = reinterpret_tensor(buf255, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf255  # reuse
    buf258 = buf228; del buf228  # reuse
    buf260 = buf226; del buf226  # reuse
    buf261 = reinterpret_tensor(buf252, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf252  # reuse
    buf262 = buf224; del buf224  # reuse
    buf259 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf263 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf264 = reinterpret_tensor(buf222, (8, 16, 196), (3136, 196, 1), 0); del buf222  # reuse
    buf265 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_53(c_void_p(buf257.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()))
    del primals_137
    buf266 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_7_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, reinterpret_tensor(primals_138, (768, 768), (1, 768), 0), out=buf266)
    buf267 = reinterpret_tensor(buf261, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf261  # reuse
    buf268 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_54(c_void_p(primals_40.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf266, (128, 196, 48), (9408, 48, 1), 0); del buf266  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf267, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf268, (128, 196, 48), (9408, 48, 1), 0), out=buf269)
    buf270 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_55(c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    buf271 = reinterpret_tensor(buf269, (1568, 768), (768, 1), 0); del buf269  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_140, buf270, reinterpret_tensor(primals_139, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf271)
    del primals_140
    buf272 = reinterpret_tensor(buf271, (8, 196, 768), (150528, 768, 1), 0); del buf271  # reuse
    buf273 = buf246; del buf246  # reuse
    buf274 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf276 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf277 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_56(c_void_p(buf272.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    del primals_42
    buf278 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf277, reinterpret_tensor(primals_141, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf278)
    del primals_142
    buf279 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_57(c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    buf280 = buf245; del buf245  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf279, reinterpret_tensor(primals_143, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf280)
    del primals_144
    buf281 = buf273; del buf273  # reuse
    buf282 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf284 = reinterpret_tensor(buf237, (8, 196, 768), (150528, 768, 1), 0); del buf237  # reuse
    buf285 = buf211; del buf211  # reuse
    cpp_fused_add_native_layer_norm_view_58(c_void_p(buf272.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del primals_44
    buf286 = buf251; del buf251  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, reinterpret_tensor(primals_145, (768, 1536), (1, 768), 0), out=buf286)
    buf287 = reinterpret_tensor(buf257, (307328, 16), (16, 1), 0); del buf257  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_146, (3, 16), (1, 3), 0), out=buf287)
    del primals_146
    buf288 = reinterpret_tensor(buf203, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf203  # reuse
    buf289 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_59(c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()))
    buf290 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf288, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf289, (128, 48, 196), (9408, 196, 1), 0), out=buf290)
    buf291 = reinterpret_tensor(buf264, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf264  # reuse
    buf292 = reinterpret_tensor(buf290, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf290  # reuse
    buf293 = buf262; del buf262  # reuse
    buf295 = buf260; del buf260  # reuse
    buf296 = reinterpret_tensor(buf287, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf287  # reuse
    buf297 = buf258; del buf258  # reuse
    buf294 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf298 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf299 = reinterpret_tensor(buf256, (8, 16, 196), (3136, 196, 1), 0); del buf256  # reuse
    buf300 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_60(c_void_p(buf292.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_147
    buf301 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_8_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, reinterpret_tensor(primals_148, (768, 768), (1, 768), 0), out=buf301)
    buf302 = reinterpret_tensor(buf296, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf296  # reuse
    buf303 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_61(c_void_p(primals_45.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf298.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf303.data_ptr()))
    buf304 = reinterpret_tensor(buf301, (128, 196, 48), (9408, 48, 1), 0); del buf301  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf302, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf303, (128, 196, 48), (9408, 48, 1), 0), out=buf304)
    buf305 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_62(c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    buf306 = reinterpret_tensor(buf304, (1568, 768), (768, 1), 0); del buf304  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_150, buf305, reinterpret_tensor(primals_149, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf306)
    del primals_150
    buf307 = buf281; del buf281  # reuse
    buf308 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf310 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf311 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_63(c_void_p(buf272.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    del primals_47
    buf312 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_152, buf311, reinterpret_tensor(primals_151, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf312)
    del primals_152
    buf313 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_64(c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_154, buf313, reinterpret_tensor(primals_153, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf314)
    del primals_154
    buf315 = buf307; del buf307  # reuse
    buf316 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf318 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf319 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_65(c_void_p(buf272.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    del primals_49
    buf320 = buf286; del buf286  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(buf319, reinterpret_tensor(primals_155, (768, 1536), (1, 768), 0), out=buf320)
    buf321 = reinterpret_tensor(buf292, (307328, 16), (16, 1), 0); del buf292  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(buf10, reinterpret_tensor(primals_156, (3, 16), (1, 3), 0), out=buf321)
    del primals_156
    buf322 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf323 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_66(c_void_p(buf320.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del buf320
    buf324 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf322, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf323, (128, 48, 196), (9408, 196, 1), 0), out=buf324)
    buf325 = reinterpret_tensor(buf299, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf299  # reuse
    buf326 = reinterpret_tensor(buf324, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf324  # reuse
    buf327 = buf297; del buf297  # reuse
    buf329 = buf295; del buf295  # reuse
    buf330 = reinterpret_tensor(buf321, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf321  # reuse
    buf331 = buf293; del buf293  # reuse
    buf328 = empty_strided((8, 16, 196, 196), (614656, 1, 3136, 16), device='cpu', dtype=torch.float32)
    buf332 = empty((8, 16, 196, 196), device='cpu', dtype=torch.float32)
    buf333 = reinterpret_tensor(buf291, (8, 16, 196), (3136, 196, 1), 0); del buf291  # reuse
    buf334 = empty_strided((8, 16, 196, 1), (3136, 1, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_unsqueeze_67(c_void_p(buf326.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    del buf325
    del buf326
    del buf327
    del buf329
    del buf331
    del buf333
    del primals_157
    buf335 = empty((1568, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_9_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf319, reinterpret_tensor(primals_158, (768, 768), (1, 768), 0), out=buf335)
    buf336 = reinterpret_tensor(buf330, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf330  # reuse
    buf337 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_mul_rsub_sigmoid_68(c_void_p(primals_50.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf337.data_ptr()))
    buf338 = reinterpret_tensor(buf335, (128, 196, 48), (9408, 48, 1), 0); del buf335  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf337, (128, 196, 48), (9408, 48, 1), 0), out=buf338)
    buf339 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_69(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf338, (1568, 768), (768, 1), 0); del buf338  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf339, reinterpret_tensor(primals_159, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf340)
    del primals_160
    buf341 = reinterpret_tensor(buf340, (8, 196, 768), (150528, 768, 1), 0); del buf340  # reuse
    buf342 = buf315; del buf315  # reuse
    buf343 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf345 = empty((8, 196, 768), device='cpu', dtype=torch.float32)
    buf346 = empty((1568, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_70(c_void_p(buf341.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf346.data_ptr()))
    del buf272
    del buf280
    del buf306
    del buf342
    del primals_52
    buf347 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_162, buf346, reinterpret_tensor(primals_161, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf347)
    del primals_162
    buf348 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_71(c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    buf349 = buf314; del buf314  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf348, reinterpret_tensor(primals_163, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf349)
    del primals_164
    buf350 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf351 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf352 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf354 = reinterpret_tensor(buf352, (8, 197, 1), (197, 1, 1), 0); del buf352  # reuse
    buf355 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_view_72(c_void_p(buf354.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf355.data_ptr()))
    del buf341
    del buf349
    del primals_2
    del primals_54
    buf356 = empty((1576, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_10_attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf355, reinterpret_tensor(primals_165, (768, 2304), (1, 768), 0), out=buf356)
    buf357 = empty((8, 16, 197, 48), device='cpu', dtype=torch.float32)
    buf358 = empty((8, 16, 48, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_73(c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    buf359 = empty((128, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf357, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf358, (128, 48, 197), (9456, 197, 1), 0), out=buf359)
    buf360 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf361 = reinterpret_tensor(buf359, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf359  # reuse
    buf362 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf363 = empty((8, 16, 197, 197), device='cpu', dtype=torch.float32)
    buf364 = empty((8, 16, 197, 48), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_74(c_void_p(buf361.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()))
    buf365 = empty((128, 197, 48), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf364, (128, 197, 48), (9456, 48, 1), 0), out=buf365)
    buf366 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_75(c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf365, (1576, 768), (768, 1), 0); del buf365  # reuse
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_167, buf366, reinterpret_tensor(primals_166, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf367)
    del primals_167
    buf368 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf369 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf371 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf372 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_76(c_void_p(buf350.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()))
    del primals_56
    buf373 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_169, buf372, reinterpret_tensor(primals_168, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf373)
    del primals_169
    buf374 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_77(c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()))
    buf375 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf374, reinterpret_tensor(primals_170, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf375)
    del primals_171
    buf376 = buf368; del buf368  # reuse
    buf377 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf379 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf380 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_78(c_void_p(buf350.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()))
    del primals_58
    buf381 = buf356; del buf356  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf380, reinterpret_tensor(primals_172, (768, 2304), (1, 768), 0), out=buf381)
    buf382 = empty((8, 16, 197, 48), device='cpu', dtype=torch.float32)
    buf383 = empty((8, 16, 48, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_79(c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    buf384 = empty((128, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf382, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf383, (128, 48, 197), (9456, 197, 1), 0), out=buf384)
    buf385 = buf360; del buf360  # reuse
    buf386 = reinterpret_tensor(buf384, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf384  # reuse
    buf387 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf388 = empty((8, 16, 197, 197), device='cpu', dtype=torch.float32)
    buf389 = empty((8, 16, 197, 48), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_80(c_void_p(buf386.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del buf381
    del buf385
    buf390 = empty((128, 197, 48), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf388, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf389, (128, 197, 48), (9456, 48, 1), 0), out=buf390)
    buf391 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_view_81(c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()))
    buf392 = reinterpret_tensor(buf390, (1576, 768), (768, 1), 0); del buf390  # reuse
    # Source Nodes: [x_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_174, buf391, reinterpret_tensor(primals_173, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf392)
    del primals_174
    buf393 = buf376; del buf376  # reuse
    buf394 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf396 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf397 = empty((1576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_82(c_void_p(buf350.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()))
    del primals_60
    buf398 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_176, buf397, reinterpret_tensor(primals_175, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf398)
    del primals_176
    buf399 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_83(c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()))
    buf400 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_171], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_178, buf399, reinterpret_tensor(primals_177, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf400)
    del primals_178
    buf401 = reinterpret_tensor(buf400, (8, 197, 768), (151296, 768, 1), 0); del buf400  # reuse
    buf402 = buf393; del buf393  # reuse
    buf403 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf405 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    buf406 = empty((8, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_84(c_void_p(buf401.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf406.data_ptr()))
    del buf367
    del buf375
    del buf392
    del buf401
    del buf402
    del primals_62
    buf407 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_180, buf406, reinterpret_tensor(primals_179, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf407)
    del primals_180
    buf408 = reinterpret_tensor(buf403, (8, 197, 1), (197, 1, 1), 0); del buf403  # reuse
    buf409 = reinterpret_tensor(buf394, (8, 197, 1), (197, 1, 1), 0); del buf394  # reuse
    buf410 = empty_strided((8, 16, 197, 197), (620944, 1, 3152, 16), device='cpu', dtype=torch.float32)
    buf411 = reinterpret_tensor(buf377, (8, 197, 1), (197, 1, 1), 0); del buf377  # reuse
    buf412 = reinterpret_tensor(buf369, (8, 197, 1), (197, 1, 1), 0); del buf369  # reuse
    buf413 = empty_strided((8, 16, 197, 197), (620944, 1, 3152, 16), device='cpu', dtype=torch.float32)
    buf414 = reinterpret_tensor(buf343, (8, 196, 1), (196, 1, 1), 0); del buf343  # reuse
    buf415 = reinterpret_tensor(buf316, (8, 196, 1), (196, 1, 1), 0); del buf316  # reuse
    buf416 = reinterpret_tensor(buf308, (8, 196, 1), (196, 1, 1), 0); del buf308  # reuse
    buf417 = reinterpret_tensor(buf282, (8, 196, 1), (196, 1, 1), 0); del buf282  # reuse
    buf418 = reinterpret_tensor(buf274, (8, 196, 1), (196, 1, 1), 0); del buf274  # reuse
    buf419 = reinterpret_tensor(buf247, (8, 196, 1), (196, 1, 1), 0); del buf247  # reuse
    buf420 = reinterpret_tensor(buf239, (8, 196, 1), (196, 1, 1), 0); del buf239  # reuse
    buf421 = reinterpret_tensor(buf213, (8, 196, 1), (196, 1, 1), 0); del buf213  # reuse
    buf422 = reinterpret_tensor(buf205, (8, 196, 1), (196, 1, 1), 0); del buf205  # reuse
    buf423 = reinterpret_tensor(buf178, (8, 196, 1), (196, 1, 1), 0); del buf178  # reuse
    buf424 = reinterpret_tensor(buf170, (8, 196, 1), (196, 1, 1), 0); del buf170  # reuse
    buf425 = reinterpret_tensor(buf144, (8, 196, 1), (196, 1, 1), 0); del buf144  # reuse
    buf426 = reinterpret_tensor(buf136, (8, 196, 1), (196, 1, 1), 0); del buf136  # reuse
    buf427 = reinterpret_tensor(buf109, (8, 196, 1), (196, 1, 1), 0); del buf109  # reuse
    buf428 = reinterpret_tensor(buf101, (8, 196, 1), (196, 1, 1), 0); del buf101  # reuse
    buf429 = reinterpret_tensor(buf75, (8, 196, 1), (196, 1, 1), 0); del buf75  # reuse
    buf430 = reinterpret_tensor(buf67, (8, 196, 1), (196, 1, 1), 0); del buf67  # reuse
    buf431 = reinterpret_tensor(buf40, (8, 196, 1), (196, 1, 1), 0); del buf40  # reuse
    buf432 = reinterpret_tensor(buf32, (8, 196, 1), (196, 1, 1), 0); del buf32  # reuse
    buf433 = reinterpret_tensor(buf4, (8, 196, 1), (196, 1, 1), 0); del buf4  # reuse
    cpp_fused__softmax_add_detach_native_layer_norm_native_layer_norm_backward_85(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf413.data_ptr()))
    return (buf407, buf7, buf7, buf7, buf7, buf7, buf7, buf7, buf7, buf7, buf7, primals_3, primals_5, primals_6, primals_8, primals_10, primals_11, primals_13, primals_15, primals_16, primals_18, primals_20, primals_21, primals_23, primals_25, primals_26, primals_28, primals_30, primals_31, primals_33, primals_35, primals_36, primals_38, primals_40, primals_41, primals_43, primals_45, primals_46, primals_48, primals_50, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, buf0, buf1, buf6, buf8, buf10, buf18, buf22, buf24, buf29, buf34, buf35, buf36, buf37, buf42, buf43, buf52, buf56, buf58, buf63, buf69, buf70, buf71, buf72, buf77, buf78, buf87, buf91, buf93, buf98, buf103, buf104, buf105, buf106, buf111, buf112, buf121, buf125, buf127, buf132, buf138, buf139, buf140, buf141, buf146, buf147, buf156, buf160, buf162, buf167, buf172, buf173, buf174, buf175, buf180, buf181, buf190, buf194, buf196, buf201, buf207, buf208, buf209, buf210, buf215, buf216, buf225, buf229, buf231, buf236, buf241, buf242, buf243, buf244, buf249, buf250, buf259, buf263, buf265, buf270, buf276, buf277, buf278, buf279, buf284, buf285, buf294, buf298, buf300, buf305, buf310, buf311, buf312, buf313, buf318, buf319, buf328, buf332, buf334, buf339, buf345, buf346, buf347, buf348, buf350, buf351, buf354, buf355, buf366, buf371, buf372, buf373, buf374, buf379, buf380, buf391, buf396, buf397, buf398, buf399, buf405, buf406, reinterpret_tensor(primals_179, (1000, 768), (768, 1), 0), buf408, reinterpret_tensor(primals_177, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_175, (3072, 768), (768, 1), 0), buf409, reinterpret_tensor(primals_173, (768, 768), (768, 1), 0), reinterpret_tensor(buf388, (128, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf389, (128, 48, 197), (9456, 1, 48), 0), buf410, reinterpret_tensor(buf382, (128, 48, 197), (9456, 1, 48), 0), reinterpret_tensor(buf383, (128, 197, 48), (9456, 1, 197), 0), reinterpret_tensor(primals_172, (2304, 768), (768, 1), 0), buf411, reinterpret_tensor(primals_170, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_168, (3072, 768), (768, 1), 0), buf412, reinterpret_tensor(primals_166, (768, 768), (768, 1), 0), reinterpret_tensor(buf363, (128, 197, 197), (38809, 1, 197), 0), reinterpret_tensor(buf364, (128, 48, 197), (9456, 1, 48), 0), buf413, reinterpret_tensor(buf357, (128, 48, 197), (9456, 1, 48), 0), reinterpret_tensor(buf358, (128, 197, 48), (9456, 1, 197), 0), reinterpret_tensor(primals_165, (2304, 768), (768, 1), 0), reinterpret_tensor(primals_163, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_161, (3072, 768), (768, 1), 0), buf414, reinterpret_tensor(primals_159, (768, 768), (768, 1), 0), reinterpret_tensor(buf336, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf337, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_158, (768, 768), (768, 1), 0), reinterpret_tensor(buf322, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf323, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_155, (1536, 768), (768, 1), 0), buf415, reinterpret_tensor(primals_153, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_151, (3072, 768), (768, 1), 0), buf416, reinterpret_tensor(primals_149, (768, 768), (768, 1), 0), reinterpret_tensor(buf302, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf303, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_148, (768, 768), (768, 1), 0), reinterpret_tensor(buf288, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf289, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_145, (1536, 768), (768, 1), 0), buf417, reinterpret_tensor(primals_143, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_141, (3072, 768), (768, 1), 0), buf418, reinterpret_tensor(primals_139, (768, 768), (768, 1), 0), reinterpret_tensor(buf267, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf268, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_138, (768, 768), (768, 1), 0), reinterpret_tensor(buf253, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf254, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_135, (1536, 768), (768, 1), 0), buf419, reinterpret_tensor(primals_133, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_131, (3072, 768), (768, 1), 0), buf420, reinterpret_tensor(primals_129, (768, 768), (768, 1), 0), reinterpret_tensor(buf233, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf234, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_128, (768, 768), (768, 1), 0), reinterpret_tensor(buf219, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf220, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_125, (1536, 768), (768, 1), 0), buf421, reinterpret_tensor(primals_123, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_121, (3072, 768), (768, 1), 0), buf422, reinterpret_tensor(primals_119, (768, 768), (768, 1), 0), reinterpret_tensor(buf198, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf199, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_118, (768, 768), (768, 1), 0), reinterpret_tensor(buf184, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf185, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_115, (1536, 768), (768, 1), 0), buf423, reinterpret_tensor(primals_113, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_111, (3072, 768), (768, 1), 0), buf424, reinterpret_tensor(primals_109, (768, 768), (768, 1), 0), reinterpret_tensor(buf164, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf165, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_108, (768, 768), (768, 1), 0), reinterpret_tensor(buf150, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf151, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_105, (1536, 768), (768, 1), 0), buf425, reinterpret_tensor(primals_103, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_101, (3072, 768), (768, 1), 0), buf426, reinterpret_tensor(primals_99, (768, 768), (768, 1), 0), reinterpret_tensor(buf129, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf130, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_98, (768, 768), (768, 1), 0), reinterpret_tensor(buf115, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf116, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_95, (1536, 768), (768, 1), 0), buf427, reinterpret_tensor(primals_93, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_91, (3072, 768), (768, 1), 0), buf428, reinterpret_tensor(primals_89, (768, 768), (768, 1), 0), reinterpret_tensor(buf95, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf96, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_88, (768, 768), (768, 1), 0), reinterpret_tensor(buf81, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf82, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_85, (1536, 768), (768, 1), 0), buf429, reinterpret_tensor(primals_83, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_81, (3072, 768), (768, 1), 0), buf430, reinterpret_tensor(primals_79, (768, 768), (768, 1), 0), reinterpret_tensor(buf60, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf61, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_78, (768, 768), (768, 1), 0), reinterpret_tensor(buf46, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf47, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_75, (1536, 768), (768, 1), 0), buf431, reinterpret_tensor(primals_73, (768, 3072), (3072, 1), 0), reinterpret_tensor(primals_71, (3072, 768), (768, 1), 0), buf432, reinterpret_tensor(primals_69, (768, 768), (768, 1), 0), reinterpret_tensor(buf26, (128, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf27, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(primals_68, (768, 768), (768, 1), 0), reinterpret_tensor(buf12, (128, 48, 196), (9408, 1, 48), 0), reinterpret_tensor(buf13, (128, 196, 48), (9408, 1, 196), 0), reinterpret_tensor(primals_65, (1536, 768), (768, 1), 0), buf433, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 768), (768, 768, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
