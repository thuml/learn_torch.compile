
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


cpp_fused_add_native_layer_norm_1 = async_compile.cpp('''
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
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
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
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_2 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
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
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
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


cpp_fused_add_native_layer_norm_9 = async_compile.cpp('''
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
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (150528L*x0)));
                    }
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_15 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_18 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_25 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_26 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_31 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_33 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_34 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_41 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_47 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused_add_native_layer_norm_49 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_50 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_56 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_57 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_58 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_63 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_64 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_65 = async_compile.cpp('''
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
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
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_72 = async_compile.cpp('''
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


cpp_fused_add_native_layer_norm_73 = async_compile.cpp('''
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
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_74 = async_compile.cpp('''
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


cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
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
                        out_ptr2[static_cast<long>(x2 + (3L*x1) + (588L*x0))] = tmp16;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115248L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (115248L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_mul_rsub_sigmoid_sum_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x2) + (3136L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp4.exp();
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3136L*x0)));
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
                                auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (3136L*x2_inner) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                                tmp_acc0_vec = tmp_acc0_vec + tmp0;
                            }
                            tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (3136L*x0)));
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = 0;
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                                tmp_acc0 = tmp_acc0 + tmp0;
                            }
                            out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))] = tmp_acc0;
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
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp10 = ([&]() { __at_align__ float tmpbuf[8]; for (long x3_inner = 0; x3_inner < 8; x3_inner++) tmpbuf[x3_inner] = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (16L*x3_inner) + (3136L*x2) + (614656L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp11 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp4 / tmp6;
                            auto tmp8 = at::vec::Vectorized<float>(tmp3);
                            auto tmp9 = tmp8 * tmp7;
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = at::vec::Vectorized<float>(tmp1);
                            auto tmp15 = tmp14 * tmp13;
                            auto tmp16 = tmp9 + tmp15;
                            tmp16.store(in_out_ptr1 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x1)];
                            auto tmp4 = in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))];
                            auto tmp5 = in_ptr3[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp8 = in_out_ptr0[static_cast<long>(x1 + (16L*x3) + (3136L*x2) + (614656L*x0))];
                            auto tmp9 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (3136L*x0))];
                            auto tmp1 = decltype(tmp0)(1) / (decltype(tmp0)(1) + std::exp(-tmp0));
                            auto tmp2 = static_cast<float>(1.0);
                            auto tmp3 = decltype(tmp2)(tmp2 - tmp1);
                            auto tmp6 = tmp4 / tmp5;
                            auto tmp7 = decltype(tmp3)(tmp3 * tmp6);
                            auto tmp10 = tmp8 / tmp9;
                            auto tmp11 = decltype(tmp1)(tmp1 * tmp10);
                            auto tmp12 = decltype(tmp7)(tmp7 + tmp11);
                            in_out_ptr1[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (614656L*x0))] = tmp12;
                        }
                    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
}
''')


cpp_fused_clone_div_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = in_ptr0[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (48L*x1) + (768L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (9408L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9408L*x2) + (150528L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (150528L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_79 = async_compile.cpp('''
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_80 = async_compile.cpp('''
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


cpp_fused_cat_native_layer_norm_81 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
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
                            auto tmp18 = [&]
                            {
                                auto tmp19 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp19;
                            }
                            ;
                            auto tmp20 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp4));
                            auto tmp21 = [&]
                            {
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr2 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                                auto tmp24 = tmp22 + tmp23;
                                return tmp24;
                            }
                            ;
                            auto tmp25 = decltype(tmp21())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp21(), to_float_mask(tmp8));
                            auto tmp26 = decltype(tmp20)::blendv(tmp25, tmp20, tmp16);
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp17);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp26);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
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
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(768.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-06);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_82 = async_compile.cpp('''
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


cpp_fused__softmax_clone_mul_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
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
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9456L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_cat_native_layer_norm_85 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                        {
                            auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
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
                            auto tmp19 = tmp17 + tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = masked_load(in_ptr0 + static_cast<long>(x2), to_float_mask(tmp4));
                                return tmp21;
                            }
                            ;
                            auto tmp22 = decltype(tmp20())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp20(), to_float_mask(tmp4));
                            auto tmp23 = [&]
                            {
                                auto tmp24 = masked_load(in_ptr1 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                                auto tmp25 = masked_load(in_ptr2 + static_cast<long>((-768L) + x2 + (768L*x1) + (150528L*x0)), to_float_mask(tmp8));
                                auto tmp26 = tmp24 + tmp25;
                                return tmp26;
                            }
                            ;
                            auto tmp27 = decltype(tmp23())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp23(), to_float_mask(tmp8));
                            auto tmp28 = decltype(tmp22)::blendv(tmp27, tmp22, tmp16);
                            auto tmp29 = tmp28 + tmp18;
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp19);
                            tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp29);
                        }
                        tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc0.mean);
                        tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = static_cast<float>(tmp_acc1.m2);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp20 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp23 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp33 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = static_cast<float>(768.0);
                        auto tmp25 = tmp23 / tmp24;
                        auto tmp26 = static_cast<float>(1e-06);
                        auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                        auto tmp28 = 1 / std::sqrt(tmp27);
                        auto tmp29 = at::vec::Vectorized<float>(tmp28);
                        auto tmp30 = tmp22 * tmp29;
                        auto tmp32 = tmp30 * tmp31;
                        auto tmp34 = tmp32 + tmp33;
                        tmp34.store(out_ptr2 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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


cpp_fused_add_cat_native_layer_norm_87 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
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
                        auto tmp19 = tmp17 + tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(in_out_ptr0 + static_cast<long>(x2 + (768L*x1) + (151296L*x0)));
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
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_88 = async_compile.cpp('''
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


cpp_fused__softmax_clone_mul_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
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
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp2;
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
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (48L*x2) + (9456L*x1) + (151296L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (48L*x1) + (9456L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (48L*x2) + (768L*x1) + (151296L*x0)));
                        }
                    }
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
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4841472L); x0+=static_cast<long>(8L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (151296L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (151296L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (151296L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(197L*x0)];
                        auto tmp8 = out_ptr1[static_cast<long>(197L*x0)];
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
                        tmp19.store(out_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 196, 768), (150528, 768, 1))
    assert_size_stride(arg1_1, (1, 1, 768), (768, 768, 1))
    assert_size_stride(arg2_1, (768, ), (1, ))
    assert_size_stride(arg3_1, (768, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (768, ), (1, ))
    assert_size_stride(arg6_1, (768, ), (1, ))
    assert_size_stride(arg7_1, (768, ), (1, ))
    assert_size_stride(arg8_1, (768, ), (1, ))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (768, ), (1, ))
    assert_size_stride(arg11_1, (768, ), (1, ))
    assert_size_stride(arg12_1, (768, ), (1, ))
    assert_size_stride(arg13_1, (768, ), (1, ))
    assert_size_stride(arg14_1, (16, ), (1, ))
    assert_size_stride(arg15_1, (768, ), (1, ))
    assert_size_stride(arg16_1, (768, ), (1, ))
    assert_size_stride(arg17_1, (768, ), (1, ))
    assert_size_stride(arg18_1, (768, ), (1, ))
    assert_size_stride(arg19_1, (16, ), (1, ))
    assert_size_stride(arg20_1, (768, ), (1, ))
    assert_size_stride(arg21_1, (768, ), (1, ))
    assert_size_stride(arg22_1, (768, ), (1, ))
    assert_size_stride(arg23_1, (768, ), (1, ))
    assert_size_stride(arg24_1, (16, ), (1, ))
    assert_size_stride(arg25_1, (768, ), (1, ))
    assert_size_stride(arg26_1, (768, ), (1, ))
    assert_size_stride(arg27_1, (768, ), (1, ))
    assert_size_stride(arg28_1, (768, ), (1, ))
    assert_size_stride(arg29_1, (16, ), (1, ))
    assert_size_stride(arg30_1, (768, ), (1, ))
    assert_size_stride(arg31_1, (768, ), (1, ))
    assert_size_stride(arg32_1, (768, ), (1, ))
    assert_size_stride(arg33_1, (768, ), (1, ))
    assert_size_stride(arg34_1, (16, ), (1, ))
    assert_size_stride(arg35_1, (768, ), (1, ))
    assert_size_stride(arg36_1, (768, ), (1, ))
    assert_size_stride(arg37_1, (768, ), (1, ))
    assert_size_stride(arg38_1, (768, ), (1, ))
    assert_size_stride(arg39_1, (16, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (16, ), (1, ))
    assert_size_stride(arg45_1, (768, ), (1, ))
    assert_size_stride(arg46_1, (768, ), (1, ))
    assert_size_stride(arg47_1, (768, ), (1, ))
    assert_size_stride(arg48_1, (768, ), (1, ))
    assert_size_stride(arg49_1, (16, ), (1, ))
    assert_size_stride(arg50_1, (768, ), (1, ))
    assert_size_stride(arg51_1, (768, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (768, ), (1, ))
    assert_size_stride(arg55_1, (768, ), (1, ))
    assert_size_stride(arg56_1, (768, ), (1, ))
    assert_size_stride(arg57_1, (768, ), (1, ))
    assert_size_stride(arg58_1, (768, ), (1, ))
    assert_size_stride(arg59_1, (768, ), (1, ))
    assert_size_stride(arg60_1, (768, ), (1, ))
    assert_size_stride(arg61_1, (768, ), (1, ))
    assert_size_stride(arg62_1, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(arg63_1, (768, ), (1, ))
    assert_size_stride(arg64_1, (1536, 768), (768, 1))
    assert_size_stride(arg65_1, (16, 3), (3, 1))
    assert_size_stride(arg66_1, (16, ), (1, ))
    assert_size_stride(arg67_1, (768, 768), (768, 1))
    assert_size_stride(arg68_1, (768, 768), (768, 1))
    assert_size_stride(arg69_1, (768, ), (1, ))
    assert_size_stride(arg70_1, (3072, 768), (768, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (768, 3072), (3072, 1))
    assert_size_stride(arg73_1, (768, ), (1, ))
    assert_size_stride(arg74_1, (1536, 768), (768, 1))
    assert_size_stride(arg75_1, (16, 3), (3, 1))
    assert_size_stride(arg76_1, (16, ), (1, ))
    assert_size_stride(arg77_1, (768, 768), (768, 1))
    assert_size_stride(arg78_1, (768, 768), (768, 1))
    assert_size_stride(arg79_1, (768, ), (1, ))
    assert_size_stride(arg80_1, (3072, 768), (768, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (768, 3072), (3072, 1))
    assert_size_stride(arg83_1, (768, ), (1, ))
    assert_size_stride(arg84_1, (1536, 768), (768, 1))
    assert_size_stride(arg85_1, (16, 3), (3, 1))
    assert_size_stride(arg86_1, (16, ), (1, ))
    assert_size_stride(arg87_1, (768, 768), (768, 1))
    assert_size_stride(arg88_1, (768, 768), (768, 1))
    assert_size_stride(arg89_1, (768, ), (1, ))
    assert_size_stride(arg90_1, (3072, 768), (768, 1))
    assert_size_stride(arg91_1, (3072, ), (1, ))
    assert_size_stride(arg92_1, (768, 3072), (3072, 1))
    assert_size_stride(arg93_1, (768, ), (1, ))
    assert_size_stride(arg94_1, (1536, 768), (768, 1))
    assert_size_stride(arg95_1, (16, 3), (3, 1))
    assert_size_stride(arg96_1, (16, ), (1, ))
    assert_size_stride(arg97_1, (768, 768), (768, 1))
    assert_size_stride(arg98_1, (768, 768), (768, 1))
    assert_size_stride(arg99_1, (768, ), (1, ))
    assert_size_stride(arg100_1, (3072, 768), (768, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (768, 3072), (3072, 1))
    assert_size_stride(arg103_1, (768, ), (1, ))
    assert_size_stride(arg104_1, (1536, 768), (768, 1))
    assert_size_stride(arg105_1, (16, 3), (3, 1))
    assert_size_stride(arg106_1, (16, ), (1, ))
    assert_size_stride(arg107_1, (768, 768), (768, 1))
    assert_size_stride(arg108_1, (768, 768), (768, 1))
    assert_size_stride(arg109_1, (768, ), (1, ))
    assert_size_stride(arg110_1, (3072, 768), (768, 1))
    assert_size_stride(arg111_1, (3072, ), (1, ))
    assert_size_stride(arg112_1, (768, 3072), (3072, 1))
    assert_size_stride(arg113_1, (768, ), (1, ))
    assert_size_stride(arg114_1, (1536, 768), (768, 1))
    assert_size_stride(arg115_1, (16, 3), (3, 1))
    assert_size_stride(arg116_1, (16, ), (1, ))
    assert_size_stride(arg117_1, (768, 768), (768, 1))
    assert_size_stride(arg118_1, (768, 768), (768, 1))
    assert_size_stride(arg119_1, (768, ), (1, ))
    assert_size_stride(arg120_1, (3072, 768), (768, 1))
    assert_size_stride(arg121_1, (3072, ), (1, ))
    assert_size_stride(arg122_1, (768, 3072), (3072, 1))
    assert_size_stride(arg123_1, (768, ), (1, ))
    assert_size_stride(arg124_1, (1536, 768), (768, 1))
    assert_size_stride(arg125_1, (16, 3), (3, 1))
    assert_size_stride(arg126_1, (16, ), (1, ))
    assert_size_stride(arg127_1, (768, 768), (768, 1))
    assert_size_stride(arg128_1, (768, 768), (768, 1))
    assert_size_stride(arg129_1, (768, ), (1, ))
    assert_size_stride(arg130_1, (3072, 768), (768, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (768, 3072), (3072, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (1536, 768), (768, 1))
    assert_size_stride(arg135_1, (16, 3), (3, 1))
    assert_size_stride(arg136_1, (16, ), (1, ))
    assert_size_stride(arg137_1, (768, 768), (768, 1))
    assert_size_stride(arg138_1, (768, 768), (768, 1))
    assert_size_stride(arg139_1, (768, ), (1, ))
    assert_size_stride(arg140_1, (3072, 768), (768, 1))
    assert_size_stride(arg141_1, (3072, ), (1, ))
    assert_size_stride(arg142_1, (768, 3072), (3072, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (1536, 768), (768, 1))
    assert_size_stride(arg145_1, (16, 3), (3, 1))
    assert_size_stride(arg146_1, (16, ), (1, ))
    assert_size_stride(arg147_1, (768, 768), (768, 1))
    assert_size_stride(arg148_1, (768, 768), (768, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (3072, 768), (768, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (768, 3072), (3072, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (1536, 768), (768, 1))
    assert_size_stride(arg155_1, (16, 3), (3, 1))
    assert_size_stride(arg156_1, (16, ), (1, ))
    assert_size_stride(arg157_1, (768, 768), (768, 1))
    assert_size_stride(arg158_1, (768, 768), (768, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (3072, 768), (768, 1))
    assert_size_stride(arg161_1, (3072, ), (1, ))
    assert_size_stride(arg162_1, (768, 3072), (3072, 1))
    assert_size_stride(arg163_1, (768, ), (1, ))
    assert_size_stride(arg164_1, (2304, 768), (768, 1))
    assert_size_stride(arg165_1, (768, 768), (768, 1))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (3072, 768), (768, 1))
    assert_size_stride(arg168_1, (3072, ), (1, ))
    assert_size_stride(arg169_1, (768, 3072), (3072, 1))
    assert_size_stride(arg170_1, (768, ), (1, ))
    assert_size_stride(arg171_1, (2304, 768), (768, 1))
    assert_size_stride(arg172_1, (768, 768), (768, 1))
    assert_size_stride(arg173_1, (768, ), (1, ))
    assert_size_stride(arg174_1, (3072, 768), (768, 1))
    assert_size_stride(arg175_1, (3072, ), (1, ))
    assert_size_stride(arg176_1, (768, 3072), (3072, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (1000, 768), (768, 1))
    assert_size_stride(arg179_1, (1000, ), (1, ))
    assert_size_stride(arg180_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((768, 3, 16, 16), (768, 1, 48, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg180_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg180_1
    del arg62_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg63_1, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (8, 768, 14, 14), (150528, 1, 10752, 768))
    del arg63_1
    del buf1
    buf3 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 196, 1), (196, 1, 1568), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf0, (8, 196, 768), (150528, 768, 1), 0); del buf0  # reuse
    cpp_fused_add_native_layer_norm_1(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf6.data_ptr()))
    del arg2_1
    del arg3_1
    buf7 = empty((1568, 1536), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf6, (1568, 768), (768, 1), 0), reinterpret_tensor(arg64_1, (768, 1536), (1, 768), 0), out=buf7)
    del arg64_1
    buf8 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf9 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_2(c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    buf10 = empty((128, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf8, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf9, (128, 48, 196), (9408, 196, 1), 0), out=buf10)
    buf11 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf12 = reinterpret_tensor(buf10, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf10  # reuse
    buf13 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf14 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf15 = empty((8, 196, 196, 3), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_3(c_void_p(buf12.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    buf16 = empty((307328, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_0_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (307328, 3), (3, 1), 0), reinterpret_tensor(arg65_1, (3, 16), (1, 3), 0), out=buf16)
    del arg65_1
    buf17 = reinterpret_tensor(buf11, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf11  # reuse
    buf18 = reinterpret_tensor(buf16, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf16  # reuse
    buf19 = empty_strided((8, 16, 196, 1), (3136, 196, 1, 25088), device='cpu', dtype=torch.float32)
    buf20 = buf12; del buf12  # reuse
    buf21 = empty((8, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_4(c_void_p(buf18.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    del arg4_1
    del arg66_1
    buf22 = reinterpret_tensor(buf9, (1568, 768), (768, 1), 0); del buf9  # reuse
    # Source Nodes: [l__mod___blocks_0_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf6, (1568, 768), (768, 1), 0), reinterpret_tensor(arg67_1, (768, 768), (1, 768), 0), out=buf22)
    del arg67_1
    buf23 = buf20; del buf20  # reuse
    buf24 = reinterpret_tensor(buf6, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf6  # reuse
    cpp_fused_clone_div_5(c_void_p(buf23.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = reinterpret_tensor(buf22, (128, 196, 48), (9408, 48, 1), 0); del buf22  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf23, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf24, (128, 196, 48), (9408, 48, 1), 0), out=buf25)
    buf26 = reinterpret_tensor(buf24, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf24  # reuse
    cpp_fused_clone_6(c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    buf27 = reinterpret_tensor(buf25, (1568, 768), (768, 1), 0); del buf25  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg69_1, reinterpret_tensor(buf26, (1568, 768), (768, 1), 0), reinterpret_tensor(arg68_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf27)
    del arg68_1
    del arg69_1
    buf28 = buf4; del buf4  # reuse
    buf29 = buf3; del buf3  # reuse
    buf31 = reinterpret_tensor(buf26, (8, 196, 768), (150528, 768, 1), 0); del buf26  # reuse
    cpp_fused_add_native_layer_norm_7(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg5_1
    del arg6_1
    buf32 = empty((1568, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf31, (1568, 768), (768, 1), 0), reinterpret_tensor(arg70_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf32)
    del arg70_1
    del arg71_1
    buf33 = reinterpret_tensor(buf32, (8, 196, 3072), (602112, 3072, 1), 0); del buf32  # reuse
    cpp_fused_gelu_8(c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf31, (1568, 768), (768, 1), 0); del buf31  # reuse
    # Source Nodes: [x_16], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg73_1, reinterpret_tensor(buf33, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg72_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf34)
    del arg72_1
    del arg73_1
    buf35 = buf29; del buf29  # reuse
    buf36 = buf28; del buf28  # reuse
    buf38 = reinterpret_tensor(buf8, (8, 196, 768), (150528, 768, 1), 0); del buf8  # reuse
    cpp_fused_add_native_layer_norm_9(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    del arg7_1
    del arg8_1
    buf39 = buf7; del buf7  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg74_1, (768, 1536), (1, 768), 0), out=buf39)
    del arg74_1
    buf40 = empty((8, 16, 196, 48), device='cpu', dtype=torch.float32)
    buf41 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_10(c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    buf42 = reinterpret_tensor(buf23, (128, 196, 196), (38416, 196, 1), 0); del buf23  # reuse
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf40, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf41, (128, 48, 196), (9408, 196, 1), 0), out=buf42)
    buf43 = reinterpret_tensor(buf21, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf21  # reuse
    buf44 = reinterpret_tensor(buf42, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf42  # reuse
    buf45 = buf19; del buf19  # reuse
    buf46 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf47 = buf15; del buf15  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_11(c_void_p(buf44.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = reinterpret_tensor(buf18, (307328, 16), (16, 1), 0); del buf18  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf47, (307328, 3), (3, 1), 0), reinterpret_tensor(arg75_1, (3, 16), (1, 3), 0), out=buf48)
    del arg75_1
    buf49 = reinterpret_tensor(buf43, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf43  # reuse
    buf50 = reinterpret_tensor(buf48, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf48  # reuse
    buf51 = reinterpret_tensor(buf17, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf17  # reuse
    buf52 = buf44; del buf44  # reuse
    buf53 = reinterpret_tensor(buf13, (8, 16, 196), (3136, 196, 1), 0); del buf13  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_12(c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg76_1
    del arg9_1
    buf54 = reinterpret_tensor(buf41, (1568, 768), (768, 1), 0); del buf41  # reuse
    # Source Nodes: [l__mod___blocks_1_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf38, (1568, 768), (768, 1), 0), reinterpret_tensor(arg77_1, (768, 768), (1, 768), 0), out=buf54)
    del arg77_1
    buf55 = buf52; del buf52  # reuse
    buf56 = reinterpret_tensor(buf38, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf38  # reuse
    cpp_fused_clone_div_13(c_void_p(buf55.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()))
    buf57 = reinterpret_tensor(buf54, (128, 196, 48), (9408, 48, 1), 0); del buf54  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf55, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf56, (128, 196, 48), (9408, 48, 1), 0), out=buf57)
    buf58 = reinterpret_tensor(buf56, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf56  # reuse
    cpp_fused_clone_14(c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()))
    buf59 = reinterpret_tensor(buf57, (1568, 768), (768, 1), 0); del buf57  # reuse
    # Source Nodes: [x_22], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg79_1, reinterpret_tensor(buf58, (1568, 768), (768, 1), 0), reinterpret_tensor(arg78_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf59)
    del arg78_1
    del arg79_1
    buf60 = reinterpret_tensor(buf59, (8, 196, 768), (150528, 768, 1), 0); del buf59  # reuse
    buf61 = buf36; del buf36  # reuse
    buf62 = buf35; del buf35  # reuse
    buf64 = reinterpret_tensor(buf58, (8, 196, 768), (150528, 768, 1), 0); del buf58  # reuse
    cpp_fused_add_native_layer_norm_15(c_void_p(buf60.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf64.data_ptr()))
    del arg0_1
    del arg10_1
    del arg11_1
    buf65 = reinterpret_tensor(buf33, (1568, 3072), (3072, 1), 0); del buf33  # reuse
    # Source Nodes: [x_26], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg81_1, reinterpret_tensor(buf64, (1568, 768), (768, 1), 0), reinterpret_tensor(arg80_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf65)
    del arg80_1
    del arg81_1
    buf66 = reinterpret_tensor(buf65, (8, 196, 3072), (602112, 3072, 1), 0); del buf65  # reuse
    cpp_fused_gelu_16(c_void_p(buf66.data_ptr()))
    buf67 = reinterpret_tensor(buf64, (1568, 768), (768, 1), 0); del buf64  # reuse
    # Source Nodes: [x_30], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg83_1, reinterpret_tensor(buf66, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg82_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf67)
    del arg82_1
    del arg83_1
    buf68 = buf62; del buf62  # reuse
    buf69 = buf61; del buf61  # reuse
    buf71 = reinterpret_tensor(buf34, (8, 196, 768), (150528, 768, 1), 0); del buf34  # reuse
    cpp_fused_add_native_layer_norm_17(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg12_1
    del arg13_1
    buf72 = buf39; del buf39  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (1568, 768), (768, 1), 0), reinterpret_tensor(arg84_1, (768, 1536), (1, 768), 0), out=buf72)
    del arg84_1
    buf73 = reinterpret_tensor(buf27, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf27  # reuse
    buf74 = reinterpret_tensor(buf2, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf2  # reuse
    cpp_fused_clone_18(c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    buf75 = reinterpret_tensor(buf55, (128, 196, 196), (38416, 196, 1), 0); del buf55  # reuse
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf73, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf74, (128, 48, 196), (9408, 196, 1), 0), out=buf75)
    buf76 = reinterpret_tensor(buf53, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf53  # reuse
    buf77 = reinterpret_tensor(buf75, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf75  # reuse
    buf78 = buf51; del buf51  # reuse
    buf79 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf80 = buf47; del buf47  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_19(c_void_p(buf77.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    buf81 = reinterpret_tensor(buf50, (307328, 16), (16, 1), 0); del buf50  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf80, (307328, 3), (3, 1), 0), reinterpret_tensor(arg85_1, (3, 16), (1, 3), 0), out=buf81)
    del arg85_1
    buf82 = reinterpret_tensor(buf76, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf76  # reuse
    buf83 = reinterpret_tensor(buf81, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf81  # reuse
    buf84 = reinterpret_tensor(buf49, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf49  # reuse
    buf85 = buf77; del buf77  # reuse
    buf86 = reinterpret_tensor(buf45, (8, 16, 196), (3136, 196, 1), 0); del buf45  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_20(c_void_p(buf83.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()))
    del arg14_1
    del arg86_1
    buf87 = reinterpret_tensor(buf74, (1568, 768), (768, 1), 0); del buf74  # reuse
    # Source Nodes: [l__mod___blocks_2_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf71, (1568, 768), (768, 1), 0), reinterpret_tensor(arg87_1, (768, 768), (1, 768), 0), out=buf87)
    del arg87_1
    buf88 = buf85; del buf85  # reuse
    buf89 = reinterpret_tensor(buf71, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf71  # reuse
    cpp_fused_clone_div_21(c_void_p(buf88.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    buf90 = reinterpret_tensor(buf87, (128, 196, 48), (9408, 48, 1), 0); del buf87  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf89, (128, 196, 48), (9408, 48, 1), 0), out=buf90)
    buf91 = reinterpret_tensor(buf89, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf89  # reuse
    cpp_fused_clone_22(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = reinterpret_tensor(buf90, (1568, 768), (768, 1), 0); del buf90  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg89_1, reinterpret_tensor(buf91, (1568, 768), (768, 1), 0), reinterpret_tensor(arg88_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf92)
    del arg88_1
    del arg89_1
    buf93 = buf69; del buf69  # reuse
    buf94 = buf68; del buf68  # reuse
    buf96 = reinterpret_tensor(buf91, (8, 196, 768), (150528, 768, 1), 0); del buf91  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg15_1
    del arg16_1
    buf97 = reinterpret_tensor(buf66, (1568, 3072), (3072, 1), 0); del buf66  # reuse
    # Source Nodes: [x_40], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg91_1, reinterpret_tensor(buf96, (1568, 768), (768, 1), 0), reinterpret_tensor(arg90_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf97)
    del arg90_1
    del arg91_1
    buf98 = reinterpret_tensor(buf97, (8, 196, 3072), (602112, 3072, 1), 0); del buf97  # reuse
    cpp_fused_gelu_24(c_void_p(buf98.data_ptr()))
    buf99 = reinterpret_tensor(buf96, (1568, 768), (768, 1), 0); del buf96  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf98, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg92_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf99)
    del arg92_1
    del arg93_1
    buf100 = buf94; del buf94  # reuse
    buf101 = buf93; del buf93  # reuse
    buf103 = reinterpret_tensor(buf73, (8, 196, 768), (150528, 768, 1), 0); del buf73  # reuse
    cpp_fused_add_native_layer_norm_25(c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf103.data_ptr()))
    del arg17_1
    del arg18_1
    buf104 = buf72; del buf72  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1568, 768), (768, 1), 0), reinterpret_tensor(arg94_1, (768, 1536), (1, 768), 0), out=buf104)
    del arg94_1
    buf105 = buf40; del buf40  # reuse
    buf106 = empty((8, 16, 48, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_26(c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    buf107 = reinterpret_tensor(buf88, (128, 196, 196), (38416, 196, 1), 0); del buf88  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf105, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf106, (128, 48, 196), (9408, 196, 1), 0), out=buf107)
    buf108 = reinterpret_tensor(buf86, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf86  # reuse
    buf109 = reinterpret_tensor(buf107, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf107  # reuse
    buf110 = buf84; del buf84  # reuse
    buf111 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf112 = buf80; del buf80  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_27(c_void_p(buf109.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()))
    buf113 = reinterpret_tensor(buf83, (307328, 16), (16, 1), 0); del buf83  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf112, (307328, 3), (3, 1), 0), reinterpret_tensor(arg95_1, (3, 16), (1, 3), 0), out=buf113)
    del arg95_1
    buf114 = reinterpret_tensor(buf108, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf108  # reuse
    buf115 = reinterpret_tensor(buf113, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf113  # reuse
    buf116 = reinterpret_tensor(buf82, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf82  # reuse
    buf117 = buf109; del buf109  # reuse
    buf118 = reinterpret_tensor(buf78, (8, 16, 196), (3136, 196, 1), 0); del buf78  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_28(c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg19_1
    del arg96_1
    buf119 = reinterpret_tensor(buf106, (1568, 768), (768, 1), 0); del buf106  # reuse
    # Source Nodes: [l__mod___blocks_3_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf103, (1568, 768), (768, 1), 0), reinterpret_tensor(arg97_1, (768, 768), (1, 768), 0), out=buf119)
    del arg97_1
    buf120 = buf117; del buf117  # reuse
    buf121 = reinterpret_tensor(buf103, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf103  # reuse
    cpp_fused_clone_div_29(c_void_p(buf120.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf121.data_ptr()))
    buf122 = reinterpret_tensor(buf119, (128, 196, 48), (9408, 48, 1), 0); del buf119  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf120, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf121, (128, 196, 48), (9408, 48, 1), 0), out=buf122)
    buf123 = reinterpret_tensor(buf121, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf121  # reuse
    cpp_fused_clone_30(c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    buf124 = reinterpret_tensor(buf122, (1568, 768), (768, 1), 0); del buf122  # reuse
    # Source Nodes: [x_50], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf123, (1568, 768), (768, 1), 0), reinterpret_tensor(arg98_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf124)
    del arg98_1
    del arg99_1
    buf125 = reinterpret_tensor(buf124, (8, 196, 768), (150528, 768, 1), 0); del buf124  # reuse
    buf126 = buf101; del buf101  # reuse
    buf127 = buf100; del buf100  # reuse
    buf129 = reinterpret_tensor(buf123, (8, 196, 768), (150528, 768, 1), 0); del buf123  # reuse
    cpp_fused_add_native_layer_norm_31(c_void_p(buf125.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg20_1
    del arg21_1
    buf130 = reinterpret_tensor(buf98, (1568, 3072), (3072, 1), 0); del buf98  # reuse
    # Source Nodes: [x_54], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg101_1, reinterpret_tensor(buf129, (1568, 768), (768, 1), 0), reinterpret_tensor(arg100_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf130)
    del arg100_1
    del arg101_1
    buf131 = reinterpret_tensor(buf130, (8, 196, 3072), (602112, 3072, 1), 0); del buf130  # reuse
    cpp_fused_gelu_32(c_void_p(buf131.data_ptr()))
    buf132 = reinterpret_tensor(buf129, (1568, 768), (768, 1), 0); del buf129  # reuse
    # Source Nodes: [x_58], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg103_1, reinterpret_tensor(buf131, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg102_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf132)
    del arg102_1
    del arg103_1
    buf133 = buf127; del buf127  # reuse
    buf134 = buf126; del buf126  # reuse
    buf136 = reinterpret_tensor(buf99, (8, 196, 768), (150528, 768, 1), 0); del buf99  # reuse
    cpp_fused_add_native_layer_norm_33(c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg22_1
    del arg23_1
    buf137 = buf104; del buf104  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (1568, 768), (768, 1), 0), reinterpret_tensor(arg104_1, (768, 1536), (1, 768), 0), out=buf137)
    del arg104_1
    buf138 = reinterpret_tensor(buf92, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf92  # reuse
    buf139 = reinterpret_tensor(buf67, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf67  # reuse
    cpp_fused_clone_34(c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()))
    buf140 = reinterpret_tensor(buf120, (128, 196, 196), (38416, 196, 1), 0); del buf120  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf138, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf139, (128, 48, 196), (9408, 196, 1), 0), out=buf140)
    buf141 = reinterpret_tensor(buf118, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf118  # reuse
    buf142 = reinterpret_tensor(buf140, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf140  # reuse
    buf143 = buf116; del buf116  # reuse
    buf144 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf145 = buf112; del buf112  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_35(c_void_p(buf142.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    buf146 = reinterpret_tensor(buf115, (307328, 16), (16, 1), 0); del buf115  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf145, (307328, 3), (3, 1), 0), reinterpret_tensor(arg105_1, (3, 16), (1, 3), 0), out=buf146)
    del arg105_1
    buf147 = reinterpret_tensor(buf141, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf141  # reuse
    buf148 = reinterpret_tensor(buf146, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf146  # reuse
    buf149 = reinterpret_tensor(buf114, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf114  # reuse
    buf150 = buf142; del buf142  # reuse
    buf151 = reinterpret_tensor(buf110, (8, 16, 196), (3136, 196, 1), 0); del buf110  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_36(c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf151.data_ptr()))
    del arg106_1
    del arg24_1
    buf152 = reinterpret_tensor(buf139, (1568, 768), (768, 1), 0); del buf139  # reuse
    # Source Nodes: [l__mod___blocks_4_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf136, (1568, 768), (768, 1), 0), reinterpret_tensor(arg107_1, (768, 768), (1, 768), 0), out=buf152)
    del arg107_1
    buf153 = buf150; del buf150  # reuse
    buf154 = reinterpret_tensor(buf136, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf136  # reuse
    cpp_fused_clone_div_37(c_void_p(buf153.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()))
    buf155 = reinterpret_tensor(buf152, (128, 196, 48), (9408, 48, 1), 0); del buf152  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf153, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf154, (128, 196, 48), (9408, 48, 1), 0), out=buf155)
    buf156 = reinterpret_tensor(buf154, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf154  # reuse
    cpp_fused_clone_38(c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    buf157 = reinterpret_tensor(buf155, (1568, 768), (768, 1), 0); del buf155  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg109_1, reinterpret_tensor(buf156, (1568, 768), (768, 1), 0), reinterpret_tensor(arg108_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf157)
    del arg108_1
    del arg109_1
    buf158 = buf134; del buf134  # reuse
    buf159 = buf133; del buf133  # reuse
    buf161 = reinterpret_tensor(buf156, (8, 196, 768), (150528, 768, 1), 0); del buf156  # reuse
    cpp_fused_add_native_layer_norm_39(c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()))
    del arg25_1
    del arg26_1
    buf162 = reinterpret_tensor(buf131, (1568, 3072), (3072, 1), 0); del buf131  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg111_1, reinterpret_tensor(buf161, (1568, 768), (768, 1), 0), reinterpret_tensor(arg110_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf162)
    del arg110_1
    del arg111_1
    buf163 = reinterpret_tensor(buf162, (8, 196, 3072), (602112, 3072, 1), 0); del buf162  # reuse
    cpp_fused_gelu_40(c_void_p(buf163.data_ptr()))
    buf164 = reinterpret_tensor(buf161, (1568, 768), (768, 1), 0); del buf161  # reuse
    # Source Nodes: [x_72], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg113_1, reinterpret_tensor(buf163, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg112_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf164)
    del arg112_1
    del arg113_1
    buf165 = buf159; del buf159  # reuse
    buf166 = buf158; del buf158  # reuse
    buf168 = reinterpret_tensor(buf138, (8, 196, 768), (150528, 768, 1), 0); del buf138  # reuse
    cpp_fused_add_native_layer_norm_41(c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()), c_void_p(buf168.data_ptr()))
    del arg27_1
    del arg28_1
    buf169 = buf137; del buf137  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (1568, 768), (768, 1), 0), reinterpret_tensor(arg114_1, (768, 1536), (1, 768), 0), out=buf169)
    del arg114_1
    buf170 = reinterpret_tensor(buf60, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf60  # reuse
    buf171 = reinterpret_tensor(buf105, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf105  # reuse
    cpp_fused_clone_42(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf153, (128, 196, 196), (38416, 196, 1), 0); del buf153  # reuse
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf170, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf171, (128, 48, 196), (9408, 196, 1), 0), out=buf172)
    buf173 = reinterpret_tensor(buf151, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf151  # reuse
    buf174 = reinterpret_tensor(buf172, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf172  # reuse
    buf175 = buf149; del buf149  # reuse
    buf176 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf177 = buf145; del buf145  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_43(c_void_p(buf174.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    buf178 = reinterpret_tensor(buf148, (307328, 16), (16, 1), 0); del buf148  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf177, (307328, 3), (3, 1), 0), reinterpret_tensor(arg115_1, (3, 16), (1, 3), 0), out=buf178)
    del arg115_1
    buf179 = reinterpret_tensor(buf173, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf173  # reuse
    buf180 = reinterpret_tensor(buf178, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf178  # reuse
    buf181 = reinterpret_tensor(buf147, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf147  # reuse
    buf182 = buf174; del buf174  # reuse
    buf183 = reinterpret_tensor(buf143, (8, 16, 196), (3136, 196, 1), 0); del buf143  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_44(c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()))
    del arg116_1
    del arg29_1
    buf184 = reinterpret_tensor(buf171, (1568, 768), (768, 1), 0); del buf171  # reuse
    # Source Nodes: [l__mod___blocks_5_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf168, (1568, 768), (768, 1), 0), reinterpret_tensor(arg117_1, (768, 768), (1, 768), 0), out=buf184)
    del arg117_1
    buf185 = buf182; del buf182  # reuse
    buf186 = reinterpret_tensor(buf168, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf168  # reuse
    cpp_fused_clone_div_45(c_void_p(buf185.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = reinterpret_tensor(buf184, (128, 196, 48), (9408, 48, 1), 0); del buf184  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf185, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf186, (128, 196, 48), (9408, 48, 1), 0), out=buf187)
    buf188 = reinterpret_tensor(buf186, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf186  # reuse
    cpp_fused_clone_46(c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()))
    buf189 = reinterpret_tensor(buf187, (1568, 768), (768, 1), 0); del buf187  # reuse
    # Source Nodes: [x_78], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf188, (1568, 768), (768, 1), 0), reinterpret_tensor(arg118_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf189)
    del arg118_1
    del arg119_1
    buf190 = reinterpret_tensor(buf189, (8, 196, 768), (150528, 768, 1), 0); del buf189  # reuse
    buf191 = buf166; del buf166  # reuse
    buf192 = buf165; del buf165  # reuse
    buf194 = reinterpret_tensor(buf188, (8, 196, 768), (150528, 768, 1), 0); del buf188  # reuse
    cpp_fused_add_native_layer_norm_47(c_void_p(buf190.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()))
    del arg30_1
    del arg31_1
    buf195 = reinterpret_tensor(buf163, (1568, 3072), (3072, 1), 0); del buf163  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf194, (1568, 768), (768, 1), 0), reinterpret_tensor(arg120_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf195)
    del arg120_1
    del arg121_1
    buf196 = reinterpret_tensor(buf195, (8, 196, 3072), (602112, 3072, 1), 0); del buf195  # reuse
    cpp_fused_gelu_48(c_void_p(buf196.data_ptr()))
    buf197 = reinterpret_tensor(buf194, (1568, 768), (768, 1), 0); del buf194  # reuse
    # Source Nodes: [x_86], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg123_1, reinterpret_tensor(buf196, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg122_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf197)
    del arg122_1
    del arg123_1
    buf198 = buf192; del buf192  # reuse
    buf199 = buf191; del buf191  # reuse
    buf201 = reinterpret_tensor(buf164, (8, 196, 768), (150528, 768, 1), 0); del buf164  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf201.data_ptr()))
    del arg32_1
    del arg33_1
    buf202 = buf169; del buf169  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (1568, 768), (768, 1), 0), reinterpret_tensor(arg124_1, (768, 1536), (1, 768), 0), out=buf202)
    del arg124_1
    buf203 = reinterpret_tensor(buf157, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf157  # reuse
    buf204 = reinterpret_tensor(buf132, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf132  # reuse
    cpp_fused_clone_50(c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()))
    buf205 = reinterpret_tensor(buf185, (128, 196, 196), (38416, 196, 1), 0); del buf185  # reuse
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf203, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf204, (128, 48, 196), (9408, 196, 1), 0), out=buf205)
    buf206 = reinterpret_tensor(buf183, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf183  # reuse
    buf207 = reinterpret_tensor(buf205, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf205  # reuse
    buf208 = buf181; del buf181  # reuse
    buf209 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf210 = buf177; del buf177  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_51(c_void_p(buf207.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(buf210.data_ptr()))
    buf211 = reinterpret_tensor(buf180, (307328, 16), (16, 1), 0); del buf180  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (307328, 3), (3, 1), 0), reinterpret_tensor(arg125_1, (3, 16), (1, 3), 0), out=buf211)
    del arg125_1
    buf212 = reinterpret_tensor(buf206, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf206  # reuse
    buf213 = reinterpret_tensor(buf211, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf211  # reuse
    buf214 = reinterpret_tensor(buf179, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf179  # reuse
    buf215 = buf207; del buf207  # reuse
    buf216 = reinterpret_tensor(buf175, (8, 16, 196), (3136, 196, 1), 0); del buf175  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_52(c_void_p(buf213.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg126_1
    del arg34_1
    buf217 = reinterpret_tensor(buf204, (1568, 768), (768, 1), 0); del buf204  # reuse
    # Source Nodes: [l__mod___blocks_6_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf201, (1568, 768), (768, 1), 0), reinterpret_tensor(arg127_1, (768, 768), (1, 768), 0), out=buf217)
    del arg127_1
    buf218 = buf215; del buf215  # reuse
    buf219 = reinterpret_tensor(buf201, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf201  # reuse
    cpp_fused_clone_div_53(c_void_p(buf218.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf217, (128, 196, 48), (9408, 48, 1), 0); del buf217  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf218, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf219, (128, 196, 48), (9408, 48, 1), 0), out=buf220)
    buf221 = reinterpret_tensor(buf219, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf219  # reuse
    cpp_fused_clone_54(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf220, (1568, 768), (768, 1), 0); del buf220  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg129_1, reinterpret_tensor(buf221, (1568, 768), (768, 1), 0), reinterpret_tensor(arg128_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf222)
    del arg128_1
    del arg129_1
    buf223 = buf199; del buf199  # reuse
    buf224 = buf198; del buf198  # reuse
    buf226 = reinterpret_tensor(buf221, (8, 196, 768), (150528, 768, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()))
    del arg35_1
    del arg36_1
    buf227 = reinterpret_tensor(buf196, (1568, 3072), (3072, 1), 0); del buf196  # reuse
    # Source Nodes: [x_96], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg131_1, reinterpret_tensor(buf226, (1568, 768), (768, 1), 0), reinterpret_tensor(arg130_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf227)
    del arg130_1
    del arg131_1
    buf228 = reinterpret_tensor(buf227, (8, 196, 3072), (602112, 3072, 1), 0); del buf227  # reuse
    cpp_fused_gelu_56(c_void_p(buf228.data_ptr()))
    buf229 = reinterpret_tensor(buf226, (1568, 768), (768, 1), 0); del buf226  # reuse
    # Source Nodes: [x_100], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg133_1, reinterpret_tensor(buf228, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf229)
    del arg132_1
    del arg133_1
    buf230 = buf224; del buf224  # reuse
    buf231 = buf223; del buf223  # reuse
    buf233 = reinterpret_tensor(buf203, (8, 196, 768), (150528, 768, 1), 0); del buf203  # reuse
    cpp_fused_add_native_layer_norm_57(c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()))
    del arg37_1
    del arg38_1
    buf234 = buf202; del buf202  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (1568, 768), (768, 1), 0), reinterpret_tensor(arg134_1, (768, 1536), (1, 768), 0), out=buf234)
    del arg134_1
    buf235 = reinterpret_tensor(buf125, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf125  # reuse
    buf236 = reinterpret_tensor(buf170, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf170  # reuse
    cpp_fused_clone_58(c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()))
    buf237 = reinterpret_tensor(buf218, (128, 196, 196), (38416, 196, 1), 0); del buf218  # reuse
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf235, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf236, (128, 48, 196), (9408, 196, 1), 0), out=buf237)
    buf238 = reinterpret_tensor(buf216, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf216  # reuse
    buf239 = reinterpret_tensor(buf237, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf237  # reuse
    buf240 = buf214; del buf214  # reuse
    buf241 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf242 = buf210; del buf210  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_59(c_void_p(buf239.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf213, (307328, 16), (16, 1), 0); del buf213  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf242, (307328, 3), (3, 1), 0), reinterpret_tensor(arg135_1, (3, 16), (1, 3), 0), out=buf243)
    del arg135_1
    buf244 = reinterpret_tensor(buf238, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf238  # reuse
    buf245 = reinterpret_tensor(buf243, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf243  # reuse
    buf246 = reinterpret_tensor(buf212, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf212  # reuse
    buf247 = buf239; del buf239  # reuse
    buf248 = reinterpret_tensor(buf208, (8, 16, 196), (3136, 196, 1), 0); del buf208  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_60(c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg136_1
    del arg39_1
    buf249 = reinterpret_tensor(buf236, (1568, 768), (768, 1), 0); del buf236  # reuse
    # Source Nodes: [l__mod___blocks_7_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf233, (1568, 768), (768, 1), 0), reinterpret_tensor(arg137_1, (768, 768), (1, 768), 0), out=buf249)
    del arg137_1
    buf250 = buf247; del buf247  # reuse
    buf251 = reinterpret_tensor(buf233, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf233  # reuse
    cpp_fused_clone_div_61(c_void_p(buf250.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = reinterpret_tensor(buf249, (128, 196, 48), (9408, 48, 1), 0); del buf249  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf250, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf251, (128, 196, 48), (9408, 48, 1), 0), out=buf252)
    buf253 = reinterpret_tensor(buf251, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf251  # reuse
    cpp_fused_clone_62(c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()))
    buf254 = reinterpret_tensor(buf252, (1568, 768), (768, 1), 0); del buf252  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg139_1, reinterpret_tensor(buf253, (1568, 768), (768, 1), 0), reinterpret_tensor(arg138_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf254)
    del arg138_1
    del arg139_1
    buf255 = reinterpret_tensor(buf254, (8, 196, 768), (150528, 768, 1), 0); del buf254  # reuse
    buf256 = buf231; del buf231  # reuse
    buf257 = buf230; del buf230  # reuse
    buf259 = reinterpret_tensor(buf253, (8, 196, 768), (150528, 768, 1), 0); del buf253  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf255.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf259.data_ptr()))
    del arg40_1
    del arg41_1
    buf260 = reinterpret_tensor(buf228, (1568, 3072), (3072, 1), 0); del buf228  # reuse
    # Source Nodes: [x_110], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf259, (1568, 768), (768, 1), 0), reinterpret_tensor(arg140_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf260)
    del arg140_1
    del arg141_1
    buf261 = reinterpret_tensor(buf260, (8, 196, 3072), (602112, 3072, 1), 0); del buf260  # reuse
    cpp_fused_gelu_64(c_void_p(buf261.data_ptr()))
    buf262 = reinterpret_tensor(buf259, (1568, 768), (768, 1), 0); del buf259  # reuse
    # Source Nodes: [x_114], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf261, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg142_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf262)
    del arg142_1
    del arg143_1
    buf263 = buf257; del buf257  # reuse
    buf264 = buf256; del buf256  # reuse
    buf266 = reinterpret_tensor(buf229, (8, 196, 768), (150528, 768, 1), 0); del buf229  # reuse
    cpp_fused_add_native_layer_norm_65(c_void_p(buf255.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del arg42_1
    del arg43_1
    buf267 = buf234; del buf234  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1568, 768), (768, 1), 0), reinterpret_tensor(arg144_1, (768, 1536), (1, 768), 0), out=buf267)
    del arg144_1
    buf268 = reinterpret_tensor(buf222, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf222  # reuse
    buf269 = reinterpret_tensor(buf197, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf197  # reuse
    cpp_fused_clone_66(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    buf270 = reinterpret_tensor(buf250, (128, 196, 196), (38416, 196, 1), 0); del buf250  # reuse
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf269, (128, 48, 196), (9408, 196, 1), 0), out=buf270)
    buf271 = reinterpret_tensor(buf248, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf248  # reuse
    buf272 = reinterpret_tensor(buf270, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf270  # reuse
    buf273 = buf246; del buf246  # reuse
    buf274 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf275 = buf242; del buf242  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_67(c_void_p(buf272.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()))
    buf276 = reinterpret_tensor(buf245, (307328, 16), (16, 1), 0); del buf245  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf275, (307328, 3), (3, 1), 0), reinterpret_tensor(arg145_1, (3, 16), (1, 3), 0), out=buf276)
    del arg145_1
    buf277 = reinterpret_tensor(buf271, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf271  # reuse
    buf278 = reinterpret_tensor(buf276, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf276  # reuse
    buf279 = reinterpret_tensor(buf244, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf244  # reuse
    buf280 = buf272; del buf272  # reuse
    buf281 = reinterpret_tensor(buf240, (8, 16, 196), (3136, 196, 1), 0); del buf240  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_68(c_void_p(buf278.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg146_1
    del arg44_1
    buf282 = reinterpret_tensor(buf269, (1568, 768), (768, 1), 0); del buf269  # reuse
    # Source Nodes: [l__mod___blocks_8_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf266, (1568, 768), (768, 1), 0), reinterpret_tensor(arg147_1, (768, 768), (1, 768), 0), out=buf282)
    del arg147_1
    buf283 = buf280; del buf280  # reuse
    buf284 = reinterpret_tensor(buf266, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf266  # reuse
    cpp_fused_clone_div_69(c_void_p(buf283.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf282, (128, 196, 48), (9408, 48, 1), 0); del buf282  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf283, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf284, (128, 196, 48), (9408, 48, 1), 0), out=buf285)
    buf286 = reinterpret_tensor(buf284, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf284  # reuse
    cpp_fused_clone_70(c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    buf287 = reinterpret_tensor(buf285, (1568, 768), (768, 1), 0); del buf285  # reuse
    # Source Nodes: [x_120], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg149_1, reinterpret_tensor(buf286, (1568, 768), (768, 1), 0), reinterpret_tensor(arg148_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf287)
    del arg148_1
    del arg149_1
    buf288 = buf264; del buf264  # reuse
    buf289 = buf263; del buf263  # reuse
    buf291 = reinterpret_tensor(buf286, (8, 196, 768), (150528, 768, 1), 0); del buf286  # reuse
    cpp_fused_add_native_layer_norm_71(c_void_p(buf255.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()))
    del arg45_1
    del arg46_1
    buf292 = reinterpret_tensor(buf261, (1568, 3072), (3072, 1), 0); del buf261  # reuse
    # Source Nodes: [x_124], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg151_1, reinterpret_tensor(buf291, (1568, 768), (768, 1), 0), reinterpret_tensor(arg150_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf292)
    del arg150_1
    del arg151_1
    buf293 = reinterpret_tensor(buf292, (8, 196, 3072), (602112, 3072, 1), 0); del buf292  # reuse
    cpp_fused_gelu_72(c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf291, (1568, 768), (768, 1), 0); del buf291  # reuse
    # Source Nodes: [x_128], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg153_1, reinterpret_tensor(buf293, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg152_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf294)
    del arg152_1
    del arg153_1
    buf295 = buf289; del buf289  # reuse
    buf296 = buf288; del buf288  # reuse
    buf298 = reinterpret_tensor(buf268, (8, 196, 768), (150528, 768, 1), 0); del buf268  # reuse
    cpp_fused_add_native_layer_norm_73(c_void_p(buf255.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del arg47_1
    del arg48_1
    buf299 = buf267; del buf267  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_qk], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (1568, 768), (768, 1), 0), reinterpret_tensor(arg154_1, (768, 1536), (1, 768), 0), out=buf299)
    del arg154_1
    buf300 = reinterpret_tensor(buf190, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf190  # reuse
    buf301 = reinterpret_tensor(buf235, (8, 16, 48, 196), (150528, 9408, 196, 1), 0); del buf235  # reuse
    cpp_fused_clone_74(c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()))
    del buf299
    buf302 = reinterpret_tensor(buf283, (128, 196, 196), (38416, 196, 1), 0); del buf283  # reuse
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf300, (128, 196, 48), (9408, 48, 1), 0), reinterpret_tensor(buf301, (128, 48, 196), (9408, 196, 1), 0), out=buf302)
    del buf300
    buf303 = reinterpret_tensor(buf281, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf281  # reuse
    buf304 = reinterpret_tensor(buf302, (8, 16, 196, 196), (614656, 38416, 196, 1), 0); del buf302  # reuse
    buf305 = buf279; del buf279  # reuse
    buf306 = empty((1, 196, 196, 3), device='cpu', dtype=torch.float32)
    buf307 = buf275; del buf275  # reuse
    cpp_fused__softmax_clone_copy_mul_select_scatter_zeros_75(c_void_p(buf304.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf278, (307328, 16), (16, 1), 0); del buf278  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_pos_proj], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf307, (307328, 3), (3, 1), 0), reinterpret_tensor(arg155_1, (3, 16), (1, 3), 0), out=buf308)
    del arg155_1
    del buf307
    buf309 = reinterpret_tensor(buf303, (8, 16, 196, 1), (3136, 1, 16, 25088), 0); del buf303  # reuse
    buf310 = reinterpret_tensor(buf308, (8, 16, 196, 196), (614656, 1, 3136, 16), 0); del buf308  # reuse
    buf311 = reinterpret_tensor(buf277, (8, 16, 196, 1), (3136, 196, 1, 25088), 0); del buf277  # reuse
    buf312 = buf304; del buf304  # reuse
    buf313 = reinterpret_tensor(buf273, (8, 16, 196), (3136, 196, 1), 0); del buf273  # reuse
    cpp_fused__softmax_add_mul_rsub_sigmoid_sum_76(c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()))
    del arg156_1
    del arg49_1
    del buf305
    del buf309
    del buf310
    del buf311
    buf314 = reinterpret_tensor(buf301, (1568, 768), (768, 1), 0); del buf301  # reuse
    # Source Nodes: [l__mod___blocks_9_attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf298, (1568, 768), (768, 1), 0), reinterpret_tensor(arg157_1, (768, 768), (1, 768), 0), out=buf314)
    del arg157_1
    buf315 = buf312; del buf312  # reuse
    buf316 = reinterpret_tensor(buf298, (8, 16, 196, 48), (150528, 9408, 48, 1), 0); del buf298  # reuse
    cpp_fused_clone_div_77(c_void_p(buf315.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf316.data_ptr()))
    del buf313
    buf317 = reinterpret_tensor(buf314, (128, 196, 48), (9408, 48, 1), 0); del buf314  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf315, (128, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf316, (128, 196, 48), (9408, 48, 1), 0), out=buf317)
    del buf315
    buf318 = reinterpret_tensor(buf316, (8, 196, 16, 48), (150528, 768, 48, 1), 0); del buf316  # reuse
    cpp_fused_clone_78(c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    buf319 = reinterpret_tensor(buf317, (1568, 768), (768, 1), 0); del buf317  # reuse
    # Source Nodes: [x_134], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf318, (1568, 768), (768, 1), 0), reinterpret_tensor(arg158_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf319)
    del arg158_1
    del arg159_1
    buf320 = reinterpret_tensor(buf319, (8, 196, 768), (150528, 768, 1), 0); del buf319  # reuse
    buf321 = buf296; del buf296  # reuse
    buf322 = buf295; del buf295  # reuse
    buf324 = reinterpret_tensor(buf318, (8, 196, 768), (150528, 768, 1), 0); del buf318  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf320.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()))
    del arg50_1
    del arg51_1
    del buf255
    del buf262
    del buf287
    del buf294
    del buf321
    del buf322
    buf325 = reinterpret_tensor(buf293, (1568, 3072), (3072, 1), 0); del buf293  # reuse
    # Source Nodes: [x_138], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg161_1, reinterpret_tensor(buf324, (1568, 768), (768, 1), 0), reinterpret_tensor(arg160_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf325)
    del arg160_1
    del arg161_1
    buf326 = reinterpret_tensor(buf325, (8, 196, 3072), (602112, 3072, 1), 0); del buf325  # reuse
    cpp_fused_gelu_80(c_void_p(buf326.data_ptr()))
    buf327 = reinterpret_tensor(buf324, (1568, 768), (768, 1), 0); del buf324  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf326, (1568, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf327)
    del arg162_1
    del arg163_1
    del buf326
    buf328 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf329 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf331 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_81(c_void_p(arg1_1.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()))
    del arg52_1
    del arg53_1
    buf332 = empty((1576, 2304), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___blocks_10_attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf331, (1576, 768), (768, 1), 0), reinterpret_tensor(arg164_1, (768, 2304), (1, 768), 0), out=buf332)
    del arg164_1
    buf333 = reinterpret_tensor(buf331, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf331  # reuse
    buf334 = empty((8, 16, 48, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    buf335 = empty((128, 197, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf333, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf334, (128, 48, 197), (9456, 197, 1), 0), out=buf335)
    buf336 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf337 = reinterpret_tensor(buf335, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf335  # reuse
    buf338 = empty_strided((8, 16, 197, 1), (3152, 197, 1, 25216), device='cpu', dtype=torch.float32)
    buf339 = buf337; del buf337  # reuse
    buf340 = reinterpret_tensor(buf334, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf334  # reuse
    cpp_fused__softmax_clone_mul_83(c_void_p(buf339.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf340.data_ptr()))
    buf341 = reinterpret_tensor(buf333, (128, 197, 48), (9456, 48, 1), 0); del buf333  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf339, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf340, (128, 197, 48), (9456, 48, 1), 0), out=buf341)
    buf342 = reinterpret_tensor(buf340, (8, 197, 16, 48), (151296, 768, 48, 1), 0); del buf340  # reuse
    cpp_fused_clone_84(c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = reinterpret_tensor(buf341, (1576, 768), (768, 1), 0); del buf341  # reuse
    # Source Nodes: [x_149], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg166_1, reinterpret_tensor(buf342, (1576, 768), (768, 1), 0), reinterpret_tensor(arg165_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf343)
    del arg165_1
    del arg166_1
    buf344 = buf329; del buf329  # reuse
    buf345 = buf328; del buf328  # reuse
    buf347 = reinterpret_tensor(buf342, (8, 197, 768), (151296, 768, 1), 0); del buf342  # reuse
    cpp_fused_add_cat_native_layer_norm_85(c_void_p(arg1_1.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del arg54_1
    del arg55_1
    buf348 = empty((1576, 3072), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_153], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg168_1, reinterpret_tensor(buf347, (1576, 768), (768, 1), 0), reinterpret_tensor(arg167_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf348)
    del arg167_1
    del arg168_1
    buf349 = reinterpret_tensor(buf348, (8, 197, 3072), (605184, 3072, 1), 0); del buf348  # reuse
    cpp_fused_gelu_86(c_void_p(buf349.data_ptr()))
    buf350 = reinterpret_tensor(buf347, (1576, 768), (768, 1), 0); del buf347  # reuse
    # Source Nodes: [x_157], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf349, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg169_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf350)
    del arg169_1
    del arg170_1
    buf351 = reinterpret_tensor(buf350, (8, 197, 768), (151296, 768, 1), 0); del buf350  # reuse
    buf352 = buf345; del buf345  # reuse
    buf353 = buf344; del buf344  # reuse
    buf355 = empty((8, 197, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_cat_native_layer_norm_87(c_void_p(buf351.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()))
    del arg1_1
    del arg56_1
    del arg57_1
    del buf320
    del buf327
    buf356 = buf332; del buf332  # reuse
    # Source Nodes: [l__mod___blocks_11_attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf355, (1576, 768), (768, 1), 0), reinterpret_tensor(arg171_1, (768, 2304), (1, 768), 0), out=buf356)
    del arg171_1
    buf357 = reinterpret_tensor(buf355, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf355  # reuse
    buf358 = reinterpret_tensor(buf343, (8, 16, 48, 197), (151296, 9456, 197, 1), 0); del buf343  # reuse
    cpp_fused_clone_88(c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()))
    buf359 = reinterpret_tensor(buf339, (128, 197, 197), (38809, 197, 1), 0); del buf339  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf357, (128, 197, 48), (9456, 48, 1), 0), reinterpret_tensor(buf358, (128, 48, 197), (9456, 197, 1), 0), out=buf359)
    buf360 = buf338; del buf338  # reuse
    buf361 = reinterpret_tensor(buf359, (8, 16, 197, 197), (620944, 38809, 197, 1), 0); del buf359  # reuse
    buf362 = buf336; del buf336  # reuse
    buf363 = buf361; del buf361  # reuse
    buf364 = reinterpret_tensor(buf358, (8, 16, 197, 48), (151296, 9456, 48, 1), 0); del buf358  # reuse
    cpp_fused__softmax_clone_mul_89(c_void_p(buf363.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf364.data_ptr()))
    del buf356
    del buf360
    del buf362
    buf365 = reinterpret_tensor(buf357, (128, 197, 48), (9456, 48, 1), 0); del buf357  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf363, (128, 197, 197), (38809, 197, 1), 0), reinterpret_tensor(buf364, (128, 197, 48), (9456, 48, 1), 0), out=buf365)
    del buf363
    buf366 = reinterpret_tensor(buf364, (8, 197, 16, 48), (151296, 768, 48, 1), 0); del buf364  # reuse
    cpp_fused_clone_90(c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    buf367 = reinterpret_tensor(buf365, (1576, 768), (768, 1), 0); del buf365  # reuse
    # Source Nodes: [x_163], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg173_1, reinterpret_tensor(buf366, (1576, 768), (768, 1), 0), reinterpret_tensor(arg172_1, (768, 768), (1, 768), 0), alpha=1, beta=1, out=buf367)
    del arg172_1
    del arg173_1
    buf368 = buf353; del buf353  # reuse
    buf369 = buf352; del buf352  # reuse
    buf371 = reinterpret_tensor(buf366, (8, 197, 768), (151296, 768, 1), 0); del buf366  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf351.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf371.data_ptr()))
    del arg58_1
    del arg59_1
    buf372 = reinterpret_tensor(buf349, (1576, 3072), (3072, 1), 0); del buf349  # reuse
    # Source Nodes: [x_167], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg175_1, reinterpret_tensor(buf371, (1576, 768), (768, 1), 0), reinterpret_tensor(arg174_1, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf372)
    del arg174_1
    del arg175_1
    buf373 = reinterpret_tensor(buf372, (8, 197, 3072), (605184, 3072, 1), 0); del buf372  # reuse
    cpp_fused_gelu_92(c_void_p(buf373.data_ptr()))
    buf374 = reinterpret_tensor(buf371, (1576, 768), (768, 1), 0); del buf371  # reuse
    # Source Nodes: [x_171], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg177_1, reinterpret_tensor(buf373, (1576, 3072), (3072, 1), 0), reinterpret_tensor(arg176_1, (3072, 768), (1, 3072), 0), alpha=1, beta=1, out=buf374)
    del arg176_1
    del arg177_1
    del buf373
    buf375 = buf369; del buf369  # reuse
    buf376 = buf368; del buf368  # reuse
    buf378 = empty((8, 768), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_native_layer_norm_93(c_void_p(buf351.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()))
    del arg60_1
    del arg61_1
    del buf351
    del buf367
    del buf374
    del buf375
    del buf376
    buf379 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_179, x_180], Original ATen: [aten.addmm, aten.clone]
    extern_kernels.addmm(arg179_1, buf378, reinterpret_tensor(arg178_1, (768, 1000), (1, 768), 0), alpha=1, beta=1, out=buf379)
    del arg178_1
    del arg179_1
    return (buf379, buf14, buf46, buf79, buf111, buf144, buf176, buf209, buf241, buf274, buf306, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 196, 768), (150528, 768, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 768), (768, 768, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((16, 3), (3, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((2304, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((3072, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((3072, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, 3072), (3072, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convit_base', benchmark_compiled_module)
