
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
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.sqrt();
                    auto tmp8 = tmp7.reciprocal();
                    auto tmp9 = static_cast<float>(1.0);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp12 = tmp2 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (1024L*x0)), static_cast<long>(64L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_addmm_avg_pool2d_native_layer_norm_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp8, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = tmp3 + tmp1;
                                auto tmp7 = tmp6 + tmp4;
                                auto tmp10 = tmp9 + tmp7;
                                auto tmp11 = static_cast<float>(0.25);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp2 = tmp1 + tmp0;
                            auto tmp4 = tmp3 + tmp2;
                            auto tmp6 = tmp5 + tmp4;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp9.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>((196L*x0) + (37632L*(c10::div_floor_integer((x1 + x1_inner), 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (1568L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                    tmp_acc0 = tmp_acc0 + tmp0;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L)) + ((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(28);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>((-1L) + (2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L)) + ((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L));
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*(static_cast<long>(x3) % static_cast<long>(3L))) + (192L*((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L)) + (384L*(static_cast<long>(x2) % static_cast<long>(14L))) + (5376L*(c10::div_floor_integer(x3, 3L))) + (5376L*((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L)) + (10752L*(c10::div_floor_integer(x2, 14L))) + (150528L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                tmp13.store(out_ptr1 + static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_col2im_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr2,
                       long* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1382400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)), static_cast<long>(288L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)));
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                        tmp0.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (1764L*x0))];
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (1764L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr2[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr3[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_8 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(1L + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(30);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>(31L + x2 + (30L*x1) + (900L*x3) + (172800L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))] = tmp13;
                        }
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(192.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_addmm_avg_pool2d_native_layer_norm_11 = async_compile.cpp('''
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
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(192.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp8, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = tmp3 + tmp1;
                                auto tmp7 = tmp6 + tmp4;
                                auto tmp10 = tmp9 + tmp7;
                                auto tmp11 = static_cast<float>(0.25);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp2 = tmp1 + tmp0;
                            auto tmp4 = tmp3 + tmp2;
                            auto tmp6 = tmp5 + tmp4;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp9.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>((196L*x0) + (37632L*(c10::div_floor_integer((x1 + x1_inner), 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (1568L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                    tmp_acc0 = tmp_acc0 + tmp0;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L)) + ((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(28);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>((-1L) + (2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L)) + ((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L));
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*(static_cast<long>(x3) % static_cast<long>(3L))) + (192L*((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L)) + (384L*(static_cast<long>(x2) % static_cast<long>(14L))) + (5376L*(c10::div_floor_integer(x3, 3L))) + (5376L*((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L)) + (10752L*(c10::div_floor_integer(x2, 14L))) + (150528L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                tmp13.store(out_ptr1 + static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_col2im_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr2,
                       long* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1382400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)), static_cast<long>(288L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)));
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                        tmp0.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (1764L*x0))];
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (1764L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr2[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr3[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(1L + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(30);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>(31L + x2 + (30L*x1) + (900L*x3) + (172800L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))] = tmp13;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp0 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = tmp6 + tmp9;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp10);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_addmm_avg_pool2d_native_layer_norm_18 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(192.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp8, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = tmp3 + tmp1;
                                auto tmp7 = tmp6 + tmp4;
                                auto tmp10 = tmp9 + tmp7;
                                auto tmp11 = static_cast<float>(0.25);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp2 = tmp1 + tmp0;
                            auto tmp4 = tmp3 + tmp2;
                            auto tmp6 = tmp5 + tmp4;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp9.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>((196L*x0) + (37632L*(c10::div_floor_integer((x1 + x1_inner), 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (1568L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                    tmp_acc0 = tmp_acc0 + tmp0;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L)) + ((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(28);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>((-1L) + (2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L)) + ((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L));
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*(static_cast<long>(x3) % static_cast<long>(3L))) + (192L*((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L)) + (384L*(static_cast<long>(x2) % static_cast<long>(14L))) + (5376L*(c10::div_floor_integer(x3, 3L))) + (5376L*((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L)) + (10752L*(c10::div_floor_integer(x2, 14L))) + (150528L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                tmp13.store(out_ptr1 + static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_col2im_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr2,
                       long* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1382400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)), static_cast<long>(288L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)));
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                        tmp0.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (1764L*x0))];
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (1764L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr2[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr3[static_cast<long>(x1 + (14L*x0))] = tmp0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(1L + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(30);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>(31L + x2 + (30L*x1) + (900L*x3) + (172800L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))] = tmp13;
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
                       const float* in_ptr5,
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 + tmp5;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp6 = tmp2 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(192.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_addmm_avg_pool2d_native_layer_norm_25 = async_compile.cpp('''
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
                       float* out_ptr3,
                       float* out_ptr4)
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp6 = tmp2 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp8);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp4 = out_ptr1[static_cast<long>(x0)];
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(192.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)), static_cast<long>(384L), tmp8, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = tmp3 + tmp1;
                                auto tmp7 = tmp6 + tmp4;
                                auto tmp10 = tmp9 + tmp7;
                                auto tmp11 = static_cast<float>(0.25);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp10 * tmp12;
                                tmp13.store(out_ptr3 + static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(8L); x3<static_cast<long>(14L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(192L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5376L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(5568L + x1 + (384L*x3) + (10752L*x2) + (150528L*x0)));
                            auto tmp2 = tmp1 + tmp0;
                            auto tmp4 = tmp3 + tmp2;
                            auto tmp6 = tmp5 + tmp4;
                            auto tmp7 = static_cast<float>(0.25);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 * tmp8;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp9.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr3[static_cast<long>(x3 + (14L*x2) + (196L*x1) + (196L*x1_inner) + (37632L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr3[static_cast<long>((196L*x0) + (37632L*(c10::div_floor_integer((x1 + x1_inner), 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(196L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr4 + static_cast<long>(x1 + (1568L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_mul_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (9L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (9L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(84672L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                    auto tmp4 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 - tmp5;
                    auto tmp7 = tmp6.exp();
                    tmp7.store(in_out_ptr0 + static_cast<long>(x1 + (9L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(8L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (9L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp1 = static_cast<float>(0.1767766952966369);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 - tmp3);
                    auto tmp5 = std::exp(tmp4);
                    in_out_ptr0[static_cast<long>(x1 + (9L*x0))] = tmp5;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            {
                                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                                float tmp_acc0 = 0;
                                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                                for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                                {
                                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                                }
                                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                                for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                                {
                                    auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                    tmp_acc0 = tmp_acc0 + tmp0;
                                }
                                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                                out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))] = static_cast<float>(tmp_acc0);
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(8L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(6L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(9L); x3+=static_cast<long>(1L))
                        {
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L)) + ((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(28);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = c10::convert<int>((-1L) + (2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L)) + ((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L));
                                auto tmp6 = tmp5 >= tmp1;
                                auto tmp7 = tmp5 < tmp3;
                                auto tmp8 = tmp2 & tmp4;
                                auto tmp9 = tmp8 & tmp6;
                                auto tmp10 = tmp9 & tmp7;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*(static_cast<long>(x3) % static_cast<long>(3L))) + (192L*((((2L*(static_cast<long>(x2) % static_cast<long>(14L))) + (static_cast<long>(x3) % static_cast<long>(3L))) >= 0L) ? 0L : 30L)) + (384L*(static_cast<long>(x2) % static_cast<long>(14L))) + (5376L*(c10::div_floor_integer(x3, 3L))) + (5376L*((((2L*(c10::div_floor_integer(x2, 14L))) + (c10::div_floor_integer(x3, 3L))) >= 0L) ? 0L : 30L)) + (10752L*(c10::div_floor_integer(x2, 14L))) + (150528L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                tmp13.store(out_ptr1 + static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_col2im_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0,
                       long* out_ptr2,
                       long* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1382400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(48L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)), static_cast<long>(288L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                tmp1.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)));
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(9L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                        tmp0.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (1764L*x0))];
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (1764L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr2[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                        out_ptr3[static_cast<long>(x1 + (14L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_29 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(1L + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(30);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>(1L + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>(31L + x2 + (30L*x1) + (900L*x3) + (172800L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))] = tmp13;
                        }
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = tmp0 + tmp3;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(192.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (768L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (768L*x0)));
                    }
                }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
                        auto tmp3 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::Vectorized<float>(tmp3);
                        auto tmp5 = tmp2 - tmp4;
                        auto tmp7 = static_cast<float>(384.0);
                        auto tmp8 = tmp6 / tmp7;
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                        auto tmp11 = 1 / std::sqrt(tmp10);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp5 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        tmp17.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp5 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp8 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 - tmp6;
                        auto tmp9 = static_cast<float>(384.0);
                        auto tmp10 = tmp8 / tmp9;
                        auto tmp11 = static_cast<float>(1e-05);
                        auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                        auto tmp13 = 1 / std::sqrt(tmp12);
                        auto tmp14 = at::vec::Vectorized<float>(tmp13);
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp15 * tmp16;
                        auto tmp19 = tmp17 + tmp18;
                        tmp19.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_gelu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (384L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                        auto tmp7 = out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp10 = out_ptr1[static_cast<long>(x1 + (196L*x0))];
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 - tmp8;
                        auto tmp11 = static_cast<float>(384.0);
                        auto tmp12 = tmp10 / tmp11;
                        auto tmp13 = static_cast<float>(1e-05);
                        auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                        auto tmp15 = 1 / std::sqrt(tmp14);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp9 * tmp16;
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp21 = tmp19 + tmp20;
                        tmp21.store(out_ptr2 + static_cast<long>(x2 + (384L*x1) + (75264L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(75264L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75264L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (75264L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (75264L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (75264L*x0)));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    tmp8.store(in_out_ptr0 + static_cast<long>(x1 + (75264L*x0)));
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_45 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_51 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_61 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_69 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_75 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_81 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_85 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_93 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_95 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_97 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_99 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_100 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_105 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_107 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_109 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp8 = out_ptr1[static_cast<long>(x0)];
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 - tmp6;
                    auto tmp9 = static_cast<float>(384.0);
                    auto tmp10 = tmp8 / tmp9;
                    auto tmp11 = static_cast<float>(1e-05);
                    auto tmp12 = decltype(tmp10)(tmp10 + tmp11);
                    auto tmp13 = 1 / std::sqrt(tmp12);
                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp15 * tmp16;
                    auto tmp19 = tmp17 + tmp18;
                    tmp19.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_111 = async_compile.cpp('''
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp10 = out_ptr1[static_cast<long>(x0)];
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 - tmp8;
                    auto tmp11 = static_cast<float>(384.0);
                    auto tmp12 = tmp10 / tmp11;
                    auto tmp13 = static_cast<float>(1e-05);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = 1 / std::sqrt(tmp14);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp19 = tmp17 * tmp18;
                    auto tmp21 = tmp19 + tmp20;
                    tmp21.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)), static_cast<long>(1152L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (1152L*x2) + (225792L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (75264L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_113 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp1 = static_cast<float>(0.1767766952966369);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp3);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (196L*x0))];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        auto tmp4 = out_ptr0[static_cast<long>(x0)];
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
                        auto tmp1 = static_cast<float>(0.1767766952966369);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(18816L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (196L*x0))] = tmp2;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(768L + x3 + (32L*x1) + (1152L*x2) + (225792L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (32L*x1) + (6272L*x2) + (75264L*x0)));
                            tmp0.store(out_ptr0 + static_cast<long>(x3 + (32L*x2) + (384L*x1) + (75264L*x0)));
                        }
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
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
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp5 = static_cast<float>(384.0);
                    auto tmp6 = tmp4 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = 1 / std::sqrt(tmp8);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_native_layer_norm_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
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
                                auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
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
                                auto tmp22 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
                                auto tmp23 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*(static_cast<long>(((-1L) + x1)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp18 = out_ptr0[static_cast<long>(x2 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x2 + (197L*x0))];
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp4));
                            return tmp6;
                        }
                        ;
                        auto tmp7 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp8 = tmp0 >= tmp3;
                        auto tmp9 = static_cast<int>(197);
                        auto tmp10 = tmp0 < tmp9;
                        auto tmp11 = [&]
                        {
                            auto tmp12 = masked_load(in_ptr1 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp13 = masked_load(in_ptr2 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp8));
                            auto tmp14 = tmp12 + tmp13;
                            return tmp14;
                        }
                        ;
                        auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp8));
                        auto tmp16 = to_float_mask(tmp4);
                        auto tmp17 = decltype(tmp7)::blendv(tmp15, tmp7, tmp16);
                        auto tmp19 = at::vec::Vectorized<float>(tmp18);
                        auto tmp20 = tmp17 - tmp19;
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = at::vec::Vectorized<float>(tmp26);
                        auto tmp28 = tmp20 * tmp27;
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp32 = tmp30 + tmp31;
                        tmp32.store(out_ptr2 + static_cast<long>(x1 + (384L*x2) + (75648L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1767766952966369);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            tmp3.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp3 = std::exp(tmp2);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp3;
                    tmp_acc0 = tmp_acc0 + tmp3;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(384L + x3 + (32L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6304L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_120 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                Welford<float> tmp_acc1 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc1_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp0 = static_cast<int>(0);
                    auto tmp1 = tmp0 >= tmp0;
                    auto tmp2 = static_cast<int>(1);
                    auto tmp3 = tmp0 < tmp2;
                    auto tmp4 = [&]
                    {
                        auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp3));
                        return tmp5;
                    }
                    ;
                    auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                    auto tmp7 = tmp0 >= tmp2;
                    auto tmp8 = static_cast<int>(197);
                    auto tmp9 = tmp0 < tmp8;
                    auto tmp10 = [&]
                    {
                        auto tmp11 = masked_load(in_ptr1 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                        auto tmp12 = masked_load(in_ptr2 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                        auto tmp13 = tmp11 + tmp12;
                        return tmp13;
                    }
                    ;
                    auto tmp14 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                    auto tmp15 = to_float_mask(tmp3);
                    auto tmp16 = decltype(tmp6)::blendv(tmp14, tmp6, tmp15);
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp3));
                        return tmp20;
                    }
                    ;
                    auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp3));
                    auto tmp22 = [&]
                    {
                        auto tmp23 = masked_load(in_ptr1 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                        auto tmp24 = masked_load(in_ptr2 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                        auto tmp25 = tmp23 + tmp24;
                        return tmp25;
                    }
                    ;
                    auto tmp26 = decltype(tmp22())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp22(), to_float_mask(tmp7));
                    auto tmp27 = decltype(tmp21)::blendv(tmp26, tmp21, tmp15);
                    auto tmp28 = tmp27 + tmp17;
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp18);
                    tmp_acc1_vec = welford_combine(tmp_acc1_vec, tmp28);
                }
                tmp_acc0 = welford_combine(tmp_acc0, welford_vec_reduce_all(tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0.mean);
                tmp_acc1 = welford_combine(tmp_acc1, welford_vec_reduce_all(tmp_acc1_vec));
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc1.m2);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (384L*x0)));
                auto tmp19 = out_ptr0[static_cast<long>(x0)];
                auto tmp22 = out_ptr1[static_cast<long>(x0)];
                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp0 = static_cast<int>(0);
                auto tmp1 = tmp0 >= tmp0;
                auto tmp2 = static_cast<int>(1);
                auto tmp3 = tmp0 < tmp2;
                auto tmp4 = [&]
                {
                    auto tmp5 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp3));
                    return tmp5;
                }
                ;
                auto tmp6 = decltype(tmp4())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp4(), to_float_mask(tmp3));
                auto tmp7 = tmp0 >= tmp2;
                auto tmp8 = static_cast<int>(197);
                auto tmp9 = tmp0 < tmp8;
                auto tmp10 = [&]
                {
                    auto tmp11 = masked_load(in_ptr1 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                    auto tmp12 = masked_load(in_ptr2 + static_cast<long>(74880L + x1 + (75264L*x0)), to_float_mask(tmp7));
                    auto tmp13 = tmp11 + tmp12;
                    return tmp13;
                }
                ;
                auto tmp14 = decltype(tmp10())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp10(), to_float_mask(tmp7));
                auto tmp15 = to_float_mask(tmp3);
                auto tmp16 = decltype(tmp6)::blendv(tmp14, tmp6, tmp15);
                auto tmp18 = tmp16 + tmp17;
                auto tmp20 = at::vec::Vectorized<float>(tmp19);
                auto tmp21 = tmp18 - tmp20;
                auto tmp23 = static_cast<float>(384.0);
                auto tmp24 = tmp22 / tmp23;
                auto tmp25 = static_cast<float>(1e-05);
                auto tmp26 = decltype(tmp24)(tmp24 + tmp25);
                auto tmp27 = 1 / std::sqrt(tmp26);
                auto tmp28 = at::vec::Vectorized<float>(tmp27);
                auto tmp29 = tmp21 * tmp28;
                auto tmp31 = tmp29 * tmp30;
                auto tmp33 = tmp31 + tmp32;
                tmp33.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_cat_native_layer_norm_122 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<int>(x2);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = c10::convert<int>(x2);
                            auto tmp7 = static_cast<int>(0);
                            auto tmp8 = tmp6 >= tmp7;
                            auto tmp9 = static_cast<int>(1);
                            auto tmp10 = tmp6 < tmp9;
                            auto tmp12 = tmp10 & tmp4;
                            auto tmp11 = [&]
                            {
                                auto tmp13 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp12));
                                return tmp13;
                            }
                            ;
                            auto tmp14 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp12));
                            auto tmp15 = tmp6 >= tmp9;
                            auto tmp16 = static_cast<int>(197);
                            auto tmp17 = tmp6 < tmp16;
                            auto tmp19 = tmp15 & tmp4;
                            auto tmp18 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr1 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp19));
                                auto tmp21 = masked_load(in_ptr2 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp19));
                                auto tmp22 = tmp20 + tmp21;
                                return tmp22;
                            }
                            ;
                            auto tmp23 = decltype(tmp18())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp18(), to_float_mask(tmp19));
                            auto tmp24 = to_float_mask(tmp10);
                            auto tmp25 = decltype(tmp14)::blendv(tmp23, tmp14, tmp24);
                            auto tmp26 = masked_load(in_ptr3 + static_cast<long>(x1 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp27 = tmp25 + tmp26;
                            auto tmp28 = masked_load(in_ptr4 + static_cast<long>(x1 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp29 = tmp27 + tmp28;
                            return tmp29;
                        }
                        ;
                        auto tmp30 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp31 = tmp0 >= tmp3;
                        auto tmp32 = static_cast<int>(197);
                        auto tmp33 = tmp0 < tmp32;
                        auto tmp34 = [&]
                        {
                            auto tmp35 = c10::convert<int>(x2);
                            auto tmp36 = static_cast<int>(0);
                            auto tmp37 = tmp35 >= tmp36;
                            auto tmp38 = static_cast<int>(1);
                            auto tmp39 = tmp35 < tmp38;
                            auto tmp41 = tmp39 & tmp31;
                            auto tmp40 = [&]
                            {
                                auto tmp42 = masked_load(in_ptr0 + static_cast<long>(x1), to_float_mask(tmp41));
                                return tmp42;
                            }
                            ;
                            auto tmp43 = decltype(tmp40())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp40(), to_float_mask(tmp41));
                            auto tmp44 = tmp35 >= tmp38;
                            auto tmp45 = static_cast<int>(197);
                            auto tmp46 = tmp35 < tmp45;
                            auto tmp48 = tmp44 & tmp31;
                            auto tmp47 = [&]
                            {
                                auto tmp49 = masked_load(in_ptr1 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp48));
                                auto tmp50 = masked_load(in_ptr2 + static_cast<long>(x1 + (384L*(static_cast<long>(((-1L) + x2)) % static_cast<long>(196L))) + (75264L*x0)), to_float_mask(tmp48));
                                auto tmp51 = tmp49 + tmp50;
                                return tmp51;
                            }
                            ;
                            auto tmp52 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp48));
                            auto tmp53 = to_float_mask(tmp39);
                            auto tmp54 = decltype(tmp43)::blendv(tmp52, tmp43, tmp53);
                            return tmp54;
                        }
                        ;
                        auto tmp55 = decltype(tmp34())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp34(), to_float_mask(tmp31));
                        auto tmp56 = to_float_mask(tmp4);
                        auto tmp57 = decltype(tmp30)::blendv(tmp55, tmp30, tmp56);
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp57.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                        #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (197L*x2) + (75648L*x0)));
                            tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                        }
                        tmp_acc0_vec.mean.store(out_ptr1 + static_cast<long>(x1 + (197L*x0)));
                        tmp_acc0_vec.m2.store(out_ptr2 + static_cast<long>(x1 + (197L*x0)));
                    }
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = out_ptr0[static_cast<long>(x1 + (197L*x2) + (75648L*x0))];
                            tmp_acc0 = welford_combine(tmp_acc0, tmp0);
                        }
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc0.mean;
                        out_ptr2[static_cast<long>(x1 + (197L*x0))] = tmp_acc0.m2;
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
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (197L*x2) + (75648L*x0)), static_cast<long>(197L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = out_ptr1[static_cast<long>(x1 + x1_inner + (197L*x0))];
                            auto tmp5 = out_ptr2[static_cast<long>(x1 + x1_inner + (197L*x0))];
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(384.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = tmp12 * tmp13;
                            auto tmp16 = tmp14 + tmp15;
                            tmp16.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75648L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x1 + (197L*x2) + (75648L*x0))];
                        auto tmp1 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp3 = out_ptr2[static_cast<long>(x1 + (197L*x0))];
                        auto tmp10 = in_ptr5[static_cast<long>(x2)];
                        auto tmp12 = in_ptr6[static_cast<long>(x2)];
                        auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                        auto tmp4 = static_cast<float>(384.0);
                        auto tmp5 = tmp3 / tmp4;
                        auto tmp6 = static_cast<float>(1e-05);
                        auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                        auto tmp8 = 1 / std::sqrt(tmp7);
                        auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                        auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                        auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                        out_ptr3[static_cast<long>(x2 + (384L*x1) + (75648L*x0))] = tmp13;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_clone_mul_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1767766952966369);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            tmp3.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (151296L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr0 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x2) + (151296L*x0)));
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (75648L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_124 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                float tmp_acc0 = -std::numeric_limits<float>::infinity();
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp0);
                }
                #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1 + (197L*x0))];
                    tmp_acc0 = max_propagate_nan(tmp_acc0, tmp0);
                }
                tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 - tmp2;
                    auto tmp4 = tmp3.exp();
                    tmp4.store(in_out_ptr0 + static_cast<long>(x1 + (197L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp4;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (197L*x0))];
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                    auto tmp3 = std::exp(tmp2);
                    in_out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp3;
                    tmp_acc0 = tmp_acc0 + tmp3;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(384L + x3 + (32L*x1) + (768L*x2) + (151296L*x0)));
                            tmp0.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6304L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_125 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (384L*x0)), static_cast<long>(384L), tmp1, 8);
                    at::vec::transpose_mxn<float,8,8>(in_ptr1 + static_cast<long>(x1 + (384L*x0)), static_cast<long>(384L), tmp1, 8);
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr0[static_cast<long>((197L*x1) + (197L*x1_inner) + (75648L*x0) + (75648L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = at::vec::Vectorized<float>::loadu(tmp1 + static_cast<long>(8L*x1_inner));
                        auto tmp3 = tmp0 + tmp2;
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp3);
                    }
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                for (long x0_inner = 0; x0_inner < 8; x0_inner++)
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((197L*x1) + (197L*x1_inner) + (75648L*x0) + (75648L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0 + x0_inner)];
                    auto tmp6 = out_ptr1[static_cast<long>(x0 + x0_inner)];
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp2 - tmp4;
                    auto tmp7 = static_cast<float>(384.0);
                    auto tmp8 = tmp6 / tmp7;
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                    auto tmp11 = 1 / std::sqrt(tmp10);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp5 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    tmp17.store(out_ptr2 + static_cast<long>(x1 + (384L*x0) + (384L*x0_inner)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_cat_clone_native_layer_norm_127 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(    welford:Welford<float>:    omp_out = welford_combine(omp_out, omp_in))     initializer(omp_priv={Welford<float>()})
                        Welford<float> tmp_acc0 = Welford<float>();
                        Welford<float> tmp_acc1 = Welford<float>();
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (75648L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (384L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                                auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x0))];
                                auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                                return tmp10;
                            }
                            ;
                            auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp12 = tmp0 >= tmp3;
                            auto tmp13 = static_cast<long>(197);
                            auto tmp14 = tmp0 < tmp13;
                            auto tmp15 = [&]
                            {
                                auto tmp16 = in_ptr0[static_cast<long>(x1 + (197L*x2) + (75648L*x0))];
                                return tmp16;
                            }
                            ;
                            auto tmp17 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                            auto tmp18 = tmp4 ? tmp11 : tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((197L*x2) + (75648L*x0))];
                                auto tmp21 = in_ptr1[static_cast<long>(x2 + (384L*x0))];
                                auto tmp22 = decltype(tmp20)(tmp20 + tmp21);
                                auto tmp23 = in_ptr2[static_cast<long>(x2 + (384L*x0))];
                                auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp4 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp26 = [&]
                            {
                                auto tmp27 = in_ptr0[static_cast<long>(x1 + (197L*x2) + (75648L*x0))];
                                return tmp27;
                            }
                            ;
                            auto tmp28 = tmp12 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                            auto tmp29 = tmp4 ? tmp25 : tmp28;
                            tmp_acc0 = welford_combine(tmp_acc0, tmp18);
                            tmp_acc1 = welford_combine(tmp_acc1, tmp29);
                        }
                        out_ptr0[static_cast<long>(x1 + (197L*x0))] = tmp_acc0.mean;
                        out_ptr1[static_cast<long>(x1 + (197L*x0))] = tmp_acc1.m2;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp19 = out_ptr0[static_cast<long>(x1 + (197L*x0))];
                        auto tmp21 = out_ptr1[static_cast<long>(x1 + (197L*x0))];
                        auto tmp28 = in_ptr3[static_cast<long>(x2)];
                        auto tmp30 = in_ptr4[static_cast<long>(x2)];
                        auto tmp0 = c10::convert<long>(x1);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>((197L*x2) + (75648L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2 + (384L*x0))];
                            auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2 + (384L*x0))];
                            auto tmp10 = decltype(tmp8)(tmp8 + tmp9);
                            return tmp10;
                        }
                        ;
                        auto tmp11 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<long>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = in_ptr0[static_cast<long>(x1 + (197L*x2) + (75648L*x0))];
                            return tmp16;
                        }
                        ;
                        auto tmp17 = tmp12 ? tmp15() : static_cast<decltype(tmp15())>(0.0);
                        auto tmp18 = tmp4 ? tmp11 : tmp17;
                        auto tmp20 = decltype(tmp18)(tmp18 - tmp19);
                        auto tmp22 = static_cast<float>(384.0);
                        auto tmp23 = tmp21 / tmp22;
                        auto tmp24 = static_cast<float>(1e-05);
                        auto tmp25 = decltype(tmp23)(tmp23 + tmp24);
                        auto tmp26 = 1 / std::sqrt(tmp25);
                        auto tmp27 = decltype(tmp20)(tmp20 * tmp26);
                        auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                        auto tmp31 = decltype(tmp29)(tmp29 + tmp30);
                        out_ptr2[static_cast<long>(x1 + (197L*x2) + (75648L*x0))] = tmp31;
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
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr2 + static_cast<long>(1L + x1 + (197L*x2) + (75648L*x0)), static_cast<long>(197L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            tmp1.store(out_ptr3 + static_cast<long>(x2 + (384L*x1) + (384L*x1_inner) + (75264L*x0)));
                        }
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(1L + x1 + (197L*x2) + (75648L*x0))];
                        out_ptr3[static_cast<long>(x2 + (384L*x1) + (75264L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_addmm_max_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1000L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1000L*x2) + (196000L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp2);
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1000L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x0_inner = 0; x0_inner < 8; x0_inner++) tmpbuf[x0_inner] = in_ptr2[static_cast<long>((197L*x0) + (197L*x0_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_mul_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 * tmp3;
            auto tmp5 = tmp0 + tmp4;
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(arg1_1, (1, 1, 384), (384, 384, 1))
    assert_size_stride(arg2_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, 192), (192, 1))
    assert_size_stride(arg16_1, (486, 192), (192, 1))
    assert_size_stride(arg17_1, (486, ), (1, ))
    assert_size_stride(arg18_1, (192, 192), (192, 1))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (576, 192), (192, 1))
    assert_size_stride(arg23_1, (576, ), (1, ))
    assert_size_stride(arg24_1, (192, 576), (576, 1))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, 192), (192, 1))
    assert_size_stride(arg29_1, (486, 192), (192, 1))
    assert_size_stride(arg30_1, (486, ), (1, ))
    assert_size_stride(arg31_1, (192, 192), (192, 1))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (576, 192), (192, 1))
    assert_size_stride(arg36_1, (576, ), (1, ))
    assert_size_stride(arg37_1, (192, 576), (576, 1))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (192, ), (1, ))
    assert_size_stride(arg41_1, (192, 192), (192, 1))
    assert_size_stride(arg42_1, (486, 192), (192, 1))
    assert_size_stride(arg43_1, (486, ), (1, ))
    assert_size_stride(arg44_1, (192, 192), (192, 1))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (576, 192), (192, 1))
    assert_size_stride(arg49_1, (576, ), (1, ))
    assert_size_stride(arg50_1, (192, 576), (576, 1))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (192, ), (1, ))
    assert_size_stride(arg53_1, (192, ), (1, ))
    assert_size_stride(arg54_1, (192, 192), (192, 1))
    assert_size_stride(arg55_1, (486, 192), (192, 1))
    assert_size_stride(arg56_1, (486, ), (1, ))
    assert_size_stride(arg57_1, (192, 192), (192, 1))
    assert_size_stride(arg58_1, (192, ), (1, ))
    assert_size_stride(arg59_1, (192, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (576, 192), (192, 1))
    assert_size_stride(arg62_1, (576, ), (1, ))
    assert_size_stride(arg63_1, (192, 576), (576, 1))
    assert_size_stride(arg64_1, (192, ), (1, ))
    assert_size_stride(arg65_1, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (1152, 384), (384, 1))
    assert_size_stride(arg70_1, (384, 384), (384, 1))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (1152, 384), (384, 1))
    assert_size_stride(arg75_1, (1152, ), (1, ))
    assert_size_stride(arg76_1, (384, 1152), (1152, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (1152, 384), (384, 1))
    assert_size_stride(arg81_1, (384, 384), (384, 1))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (1152, 384), (384, 1))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (384, 1152), (1152, 1))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (1152, 384), (384, 1))
    assert_size_stride(arg92_1, (384, 384), (384, 1))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (1152, 384), (384, 1))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (384, 1152), (1152, 1))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (1152, 384), (384, 1))
    assert_size_stride(arg103_1, (384, 384), (384, 1))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (1152, 384), (384, 1))
    assert_size_stride(arg108_1, (1152, ), (1, ))
    assert_size_stride(arg109_1, (384, 1152), (1152, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (1152, 384), (384, 1))
    assert_size_stride(arg114_1, (384, 384), (384, 1))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (1152, 384), (384, 1))
    assert_size_stride(arg119_1, (1152, ), (1, ))
    assert_size_stride(arg120_1, (384, 1152), (1152, 1))
    assert_size_stride(arg121_1, (384, ), (1, ))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, ), (1, ))
    assert_size_stride(arg124_1, (1152, 384), (384, 1))
    assert_size_stride(arg125_1, (384, 384), (384, 1))
    assert_size_stride(arg126_1, (384, ), (1, ))
    assert_size_stride(arg127_1, (384, ), (1, ))
    assert_size_stride(arg128_1, (384, ), (1, ))
    assert_size_stride(arg129_1, (1152, 384), (384, 1))
    assert_size_stride(arg130_1, (1152, ), (1, ))
    assert_size_stride(arg131_1, (384, 1152), (1152, 1))
    assert_size_stride(arg132_1, (384, ), (1, ))
    assert_size_stride(arg133_1, (384, ), (1, ))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (1152, 384), (384, 1))
    assert_size_stride(arg136_1, (384, 384), (384, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, ), (1, ))
    assert_size_stride(arg139_1, (384, ), (1, ))
    assert_size_stride(arg140_1, (1152, 384), (384, 1))
    assert_size_stride(arg141_1, (1152, ), (1, ))
    assert_size_stride(arg142_1, (384, 1152), (1152, 1))
    assert_size_stride(arg143_1, (384, ), (1, ))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (1152, 384), (384, 1))
    assert_size_stride(arg147_1, (384, 384), (384, 1))
    assert_size_stride(arg148_1, (384, ), (1, ))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (1152, 384), (384, 1))
    assert_size_stride(arg152_1, (1152, ), (1, ))
    assert_size_stride(arg153_1, (384, 1152), (1152, 1))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (1152, 384), (384, 1))
    assert_size_stride(arg158_1, (384, 384), (384, 1))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (384, ), (1, ))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (1152, 384), (384, 1))
    assert_size_stride(arg163_1, (1152, ), (1, ))
    assert_size_stride(arg164_1, (384, 1152), (1152, 1))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (1152, 384), (384, 1))
    assert_size_stride(arg169_1, (384, 384), (384, 1))
    assert_size_stride(arg170_1, (384, ), (1, ))
    assert_size_stride(arg171_1, (384, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (1152, 384), (384, 1))
    assert_size_stride(arg174_1, (1152, ), (1, ))
    assert_size_stride(arg175_1, (384, 1152), (1152, 1))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (384, ), (1, ))
    assert_size_stride(arg179_1, (1152, 384), (384, 1))
    assert_size_stride(arg180_1, (384, 384), (384, 1))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (1152, 384), (384, 1))
    assert_size_stride(arg185_1, (1152, ), (1, ))
    assert_size_stride(arg186_1, (384, 1152), (1152, 1))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (384, ), (1, ))
    assert_size_stride(arg190_1, (1152, 384), (384, 1))
    assert_size_stride(arg191_1, (384, 384), (384, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (384, ), (1, ))
    assert_size_stride(arg195_1, (1152, 384), (384, 1))
    assert_size_stride(arg196_1, (1152, ), (1, ))
    assert_size_stride(arg197_1, (384, 1152), (1152, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (1152, 384), (384, 1))
    assert_size_stride(arg202_1, (384, 384), (384, 1))
    assert_size_stride(arg203_1, (384, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (1152, 384), (384, 1))
    assert_size_stride(arg207_1, (1152, ), (1, ))
    assert_size_stride(arg208_1, (384, 1152), (1152, 1))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (1152, 384), (384, 1))
    assert_size_stride(arg213_1, (384, 384), (384, 1))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (1152, 384), (384, 1))
    assert_size_stride(arg218_1, (1152, ), (1, ))
    assert_size_stride(arg219_1, (384, 1152), (1152, 1))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (768, 384), (384, 1))
    assert_size_stride(arg224_1, (384, 384), (384, 1))
    assert_size_stride(arg225_1, (384, 384), (384, 1))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, ), (1, ))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (1152, 384), (384, 1))
    assert_size_stride(arg230_1, (1152, ), (1, ))
    assert_size_stride(arg231_1, (384, 1152), (1152, 1))
    assert_size_stride(arg232_1, (384, ), (1, ))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (768, 384), (384, 1))
    assert_size_stride(arg236_1, (384, 384), (384, 1))
    assert_size_stride(arg237_1, (384, 384), (384, 1))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (1152, 384), (384, 1))
    assert_size_stride(arg242_1, (1152, ), (1, ))
    assert_size_stride(arg243_1, (384, 1152), (1152, 1))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (1000, 384), (384, 1))
    assert_size_stride(arg248_1, (1000, ), (1, ))
    assert_size_stride(arg249_1, (1000, 384), (384, 1))
    assert_size_stride(arg250_1, (1000, ), (1, ))
    assert_size_stride(arg251_1, (64, ), (1, ))
    assert_size_stride(arg252_1, (64, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (64, ), (1, ))
    assert_size_stride(arg255_1, (64, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (64, ), (1, ))
    assert_size_stride(arg258_1, (64, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg260_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg260_1
    del arg2_1
    # Source Nodes: [l__mod___patch_embed_conv_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg251_1
    del arg252_1
    del arg3_1
    del arg4_1
    del arg5_1
    # Source Nodes: [l__mod___patch_embed_conv_1, l__mod___patch_embed_conv_2, l__mod___patch_embed_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf3
    buf6 = buf5; del buf5  # reuse
    buf7 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg254_1
    del arg255_1
    del arg6_1
    del arg7_1
    del arg8_1
    # Source Nodes: [l__mod___patch_embed_conv_4, l__mod___patch_embed_conv_5, l__mod___patch_embed_conv_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((192, 64, 4, 4), (1024, 1, 256, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg10_1
    del arg11_1
    del arg257_1
    del arg258_1
    del arg9_1
    # Source Nodes: [l__mod___patch_embed_conv_7, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf11 = extern_kernels.convolution(buf9, buf10, arg12_1, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf11, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg12_1
    del buf10
    del buf9
    buf12 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf15 = reinterpret_tensor(buf0, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf0  # reuse
    buf16 = empty((8, 192, 14, 14), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((1568, 192), (1, 1568), device='cpu', dtype=torch.float32)
    cpp_fused_addmm_avg_pool2d_native_layer_norm_4(c_void_p(buf11.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del arg13_1
    del arg14_1
    buf18 = empty((1568, 486), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___0___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg17_1, buf17, reinterpret_tensor(arg16_1, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf18)
    del arg16_1
    del arg17_1
    buf19 = empty_strided((8, 6, 196, 9, 1), (10584, 9, 54, 1, 84672), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf18, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf18  # reuse
    buf21 = empty_strided((8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_mul_5(c_void_p(buf20.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf21.data_ptr()))
    buf22 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf15, (6272, 192), (192, 1), 0), reinterpret_tensor(arg15_1, (192, 192), (1, 192), 0), out=buf22)
    del arg15_1
    buf23 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf24 = empty((8, 6, 196, 9, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_6(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()))
    buf25 = empty((9408, 9, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf23, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf24, (9408, 9, 32), (288, 32, 1), 0), out=buf25)
    buf26 = empty((8, 192, 30, 30), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf24, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf24  # reuse
    buf28 = reinterpret_tensor(buf27, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf27  # reuse
    buf29 = empty((3, 14), device='cpu', dtype=torch.int64)
    buf30 = empty((3, 14), device='cpu', dtype=torch.int64)
    cpp_fused_clone_col2im_7(c_void_p(buf28.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    aten.index_put_(buf26, [None, None, reinterpret_tensor(buf29, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf30], buf28, True)
    buf33 = reinterpret_tensor(buf22, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf22  # reuse
    cpp_fused_clone_8(c_void_p(buf26.data_ptr()), c_void_p(buf33.data_ptr()))
    buf34 = reinterpret_tensor(buf15, (6272, 192), (192, 1), 0); del buf15  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf33, (6272, 192), (192, 1), 0), reinterpret_tensor(arg18_1, (192, 192), (1, 192), 0), out=buf34)
    del arg18_1
    buf35 = buf13; del buf13  # reuse
    buf36 = buf12; del buf12  # reuse
    buf38 = buf33; del buf33  # reuse
    cpp_fused_add_native_layer_norm_9(c_void_p(buf11.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()))
    del arg20_1
    del arg21_1
    buf39 = empty((6272, 576), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg23_1, reinterpret_tensor(buf38, (6272, 192), (192, 1), 0), reinterpret_tensor(arg22_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf39)
    del arg22_1
    del arg23_1
    buf40 = reinterpret_tensor(buf39, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf39  # reuse
    cpp_fused_gelu_10(c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf38, (6272, 192), (192, 1), 0); del buf38  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg25_1, reinterpret_tensor(buf40, (6272, 576), (576, 1), 0), reinterpret_tensor(arg24_1, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf41)
    del arg24_1
    del arg25_1
    buf42 = buf36; del buf36  # reuse
    buf43 = buf35; del buf35  # reuse
    buf45 = empty((8, 28, 28, 192), device='cpu', dtype=torch.float32)
    buf46 = reinterpret_tensor(buf17, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf17  # reuse
    buf47 = reinterpret_tensor(buf16, (1568, 192), (1, 1568), 0); del buf16  # reuse
    cpp_fused_add_addmm_avg_pool2d_native_layer_norm_11(c_void_p(buf11.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg26_1
    del arg27_1
    buf48 = reinterpret_tensor(buf23, (1568, 486), (486, 1), 0); del buf23  # reuse
    # Source Nodes: [getattr_l__mod___network_0___1___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg30_1, buf47, reinterpret_tensor(arg29_1, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf48)
    del arg29_1
    del arg30_1
    buf49 = reinterpret_tensor(buf21, (8, 6, 196, 9, 1), (10584, 9, 54, 1, 84672), 0); del buf21  # reuse
    buf50 = reinterpret_tensor(buf48, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf48  # reuse
    buf51 = reinterpret_tensor(buf19, (8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), 0); del buf19  # reuse
    cpp_fused__softmax_mul_12(c_void_p(buf50.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf51.data_ptr()))
    buf52 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___1___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf45, (6272, 192), (192, 1), 0), reinterpret_tensor(arg28_1, (192, 192), (1, 192), 0), out=buf52)
    del arg28_1
    buf53 = reinterpret_tensor(buf20, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf20  # reuse
    buf54 = reinterpret_tensor(buf28, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf28  # reuse
    cpp_fused__softmax_clone_13(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    buf55 = buf25; del buf25  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf53, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf54, (9408, 9, 32), (288, 32, 1), 0), out=buf55)
    buf56 = buf26; del buf26  # reuse
    buf57 = reinterpret_tensor(buf54, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf54  # reuse
    buf58 = reinterpret_tensor(buf57, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf57  # reuse
    buf59 = buf30; del buf30  # reuse
    buf60 = buf29; del buf29  # reuse
    cpp_fused_clone_col2im_14(c_void_p(buf58.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()))
    aten.index_put_(buf56, [None, None, reinterpret_tensor(buf59, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf60], buf58, True)
    buf63 = reinterpret_tensor(buf52, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf52  # reuse
    cpp_fused_clone_15(c_void_p(buf56.data_ptr()), c_void_p(buf63.data_ptr()))
    buf64 = reinterpret_tensor(buf45, (6272, 192), (192, 1), 0); del buf45  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf63, (6272, 192), (192, 1), 0), reinterpret_tensor(arg31_1, (192, 192), (1, 192), 0), out=buf64)
    del arg31_1
    buf65 = reinterpret_tensor(buf64, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf64  # reuse
    buf66 = buf43; del buf43  # reuse
    buf67 = buf42; del buf42  # reuse
    buf69 = buf63; del buf63  # reuse
    cpp_fused_add_native_layer_norm_16(c_void_p(buf65.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()))
    del arg19_1
    del arg32_1
    del arg33_1
    del arg34_1
    buf70 = reinterpret_tensor(buf40, (6272, 576), (576, 1), 0); del buf40  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg36_1, reinterpret_tensor(buf69, (6272, 192), (192, 1), 0), reinterpret_tensor(arg35_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf70)
    del arg35_1
    del arg36_1
    buf71 = reinterpret_tensor(buf70, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf70  # reuse
    cpp_fused_gelu_17(c_void_p(buf71.data_ptr()))
    buf72 = reinterpret_tensor(buf69, (6272, 192), (192, 1), 0); del buf69  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg38_1, reinterpret_tensor(buf71, (6272, 576), (576, 1), 0), reinterpret_tensor(arg37_1, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf72)
    del arg37_1
    del arg38_1
    buf73 = buf67; del buf67  # reuse
    buf74 = buf66; del buf66  # reuse
    buf76 = reinterpret_tensor(buf41, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf41  # reuse
    buf77 = reinterpret_tensor(buf47, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf47  # reuse
    buf78 = reinterpret_tensor(buf46, (1568, 192), (1, 1568), 0); del buf46  # reuse
    cpp_fused_add_addmm_avg_pool2d_native_layer_norm_18(c_void_p(buf65.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg39_1
    del arg40_1
    buf79 = reinterpret_tensor(buf53, (1568, 486), (486, 1), 0); del buf53  # reuse
    # Source Nodes: [getattr_l__mod___network_0___2___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg43_1, buf78, reinterpret_tensor(arg42_1, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf79)
    del arg42_1
    del arg43_1
    buf80 = reinterpret_tensor(buf51, (8, 6, 196, 9, 1), (10584, 9, 54, 1, 84672), 0); del buf51  # reuse
    buf81 = reinterpret_tensor(buf79, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf79  # reuse
    buf82 = reinterpret_tensor(buf49, (8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), 0); del buf49  # reuse
    cpp_fused__softmax_mul_19(c_void_p(buf81.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = buf34; del buf34  # reuse
    # Source Nodes: [getattr_l__mod___network_0___2___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (6272, 192), (192, 1), 0), reinterpret_tensor(arg41_1, (192, 192), (1, 192), 0), out=buf83)
    del arg41_1
    buf84 = reinterpret_tensor(buf50, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf50  # reuse
    buf85 = reinterpret_tensor(buf58, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf58  # reuse
    cpp_fused__softmax_clone_20(c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = buf55; del buf55  # reuse
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf84, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf85, (9408, 9, 32), (288, 32, 1), 0), out=buf86)
    buf87 = buf56; del buf56  # reuse
    buf88 = reinterpret_tensor(buf85, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf85  # reuse
    buf89 = reinterpret_tensor(buf88, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf88  # reuse
    buf90 = buf60; del buf60  # reuse
    buf91 = buf59; del buf59  # reuse
    cpp_fused_clone_col2im_21(c_void_p(buf89.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    aten.index_put_(buf87, [None, None, reinterpret_tensor(buf90, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf91], buf89, True)
    buf94 = reinterpret_tensor(buf83, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf83  # reuse
    cpp_fused_clone_22(c_void_p(buf87.data_ptr()), c_void_p(buf94.data_ptr()))
    buf95 = reinterpret_tensor(buf76, (6272, 192), (192, 1), 0); del buf76  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (6272, 192), (192, 1), 0), reinterpret_tensor(arg44_1, (192, 192), (1, 192), 0), out=buf95)
    del arg44_1
    buf96 = buf74; del buf74  # reuse
    buf97 = buf73; del buf73  # reuse
    buf99 = buf94; del buf94  # reuse
    cpp_fused_add_native_layer_norm_23(c_void_p(buf65.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()))
    del arg46_1
    del arg47_1
    buf100 = reinterpret_tensor(buf71, (6272, 576), (576, 1), 0); del buf71  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg49_1, reinterpret_tensor(buf99, (6272, 192), (192, 1), 0), reinterpret_tensor(arg48_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf100)
    del arg48_1
    del arg49_1
    buf101 = reinterpret_tensor(buf100, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf100  # reuse
    cpp_fused_gelu_24(c_void_p(buf101.data_ptr()))
    buf102 = reinterpret_tensor(buf99, (6272, 192), (192, 1), 0); del buf99  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg51_1, reinterpret_tensor(buf101, (6272, 576), (576, 1), 0), reinterpret_tensor(arg50_1, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf102)
    del arg50_1
    del arg51_1
    buf103 = reinterpret_tensor(buf102, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf102  # reuse
    buf104 = buf97; del buf97  # reuse
    buf105 = buf96; del buf96  # reuse
    buf107 = reinterpret_tensor(buf11, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf11  # reuse
    buf108 = reinterpret_tensor(buf78, (8, 192, 14, 14), (37632, 196, 14, 1), 0); del buf78  # reuse
    buf109 = reinterpret_tensor(buf77, (1568, 192), (1, 1568), 0); del buf77  # reuse
    cpp_fused_add_addmm_avg_pool2d_native_layer_norm_25(c_void_p(buf103.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()))
    del arg45_1
    del arg52_1
    del arg53_1
    del buf108
    del buf65
    del buf72
    buf110 = reinterpret_tensor(buf84, (1568, 486), (486, 1), 0); del buf84  # reuse
    # Source Nodes: [getattr_l__mod___network_0___3___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg56_1, buf109, reinterpret_tensor(arg55_1, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf110)
    del arg55_1
    del arg56_1
    del buf109
    buf111 = reinterpret_tensor(buf82, (8, 6, 196, 9, 1), (10584, 9, 54, 1, 84672), 0); del buf82  # reuse
    buf112 = reinterpret_tensor(buf110, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf110  # reuse
    buf113 = reinterpret_tensor(buf80, (8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), 0); del buf80  # reuse
    cpp_fused__softmax_mul_26(c_void_p(buf112.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()))
    del buf111
    buf114 = buf95; del buf95  # reuse
    # Source Nodes: [getattr_l__mod___network_0___3___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf107, (6272, 192), (192, 1), 0), reinterpret_tensor(arg54_1, (192, 192), (1, 192), 0), out=buf114)
    del arg54_1
    buf115 = reinterpret_tensor(buf81, (8, 6, 196, 9, 9), (95256, 15876, 81, 9, 1), 0); del buf81  # reuse
    buf116 = reinterpret_tensor(buf89, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf89  # reuse
    cpp_fused__softmax_clone_27(c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf112
    del buf113
    buf117 = buf86; del buf86  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf115, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf116, (9408, 9, 32), (288, 32, 1), 0), out=buf117)
    del buf115
    buf118 = buf87; del buf87  # reuse
    buf119 = reinterpret_tensor(buf116, (8, 6, 32, 9, 196), (338688, 56448, 1764, 196, 1), 0); del buf116  # reuse
    buf120 = reinterpret_tensor(buf119, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf119  # reuse
    buf121 = buf91; del buf91  # reuse
    buf122 = buf90; del buf90  # reuse
    cpp_fused_clone_col2im_28(c_void_p(buf120.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf117
    aten.index_put_(buf118, [None, None, reinterpret_tensor(buf121, (3, 14, 1, 1), (14, 1, 0, 0), 0), buf122], buf120, True)
    del buf120
    del buf121
    del buf122
    buf125 = reinterpret_tensor(buf114, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf114  # reuse
    cpp_fused_clone_29(c_void_p(buf118.data_ptr()), c_void_p(buf125.data_ptr()))
    del buf118
    buf126 = reinterpret_tensor(buf107, (6272, 192), (192, 1), 0); del buf107  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf125, (6272, 192), (192, 1), 0), reinterpret_tensor(arg57_1, (192, 192), (1, 192), 0), out=buf126)
    del arg57_1
    buf127 = buf105; del buf105  # reuse
    buf128 = buf104; del buf104  # reuse
    buf130 = buf125; del buf125  # reuse
    cpp_fused_add_native_layer_norm_30(c_void_p(buf103.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg59_1
    del arg60_1
    del buf127
    del buf128
    buf131 = reinterpret_tensor(buf101, (6272, 576), (576, 1), 0); del buf101  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg62_1, reinterpret_tensor(buf130, (6272, 192), (192, 1), 0), reinterpret_tensor(arg61_1, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf131)
    del arg61_1
    del arg62_1
    buf132 = reinterpret_tensor(buf131, (8, 28, 28, 576), (451584, 16128, 576, 1), 0); del buf131  # reuse
    cpp_fused_gelu_31(c_void_p(buf132.data_ptr()))
    buf133 = reinterpret_tensor(buf130, (6272, 192), (192, 1), 0); del buf130  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg64_1, reinterpret_tensor(buf132, (6272, 576), (576, 1), 0), reinterpret_tensor(arg63_1, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf133)
    del arg63_1
    del arg64_1
    del buf132
    buf134 = reinterpret_tensor(buf133, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf133  # reuse
    buf135 = empty_strided((384, 192, 2, 2), (768, 1, 384, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_32(c_void_p(buf134.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(buf135.data_ptr()))
    del arg58_1
    del arg65_1
    del buf103
    del buf126
    # Source Nodes: [x_53], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(reinterpret_tensor(buf134, (8, 192, 28, 28), (150528, 1, 5376, 192), 0), buf135, arg66_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf136, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg66_1
    del buf134
    del buf135
    buf137 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf138 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf140 = empty((8, 14, 14, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_33(c_void_p(buf136.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf140.data_ptr()))
    del arg67_1
    del arg68_1
    buf141 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_2___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (1568, 384), (384, 1), 0), reinterpret_tensor(arg69_1, (384, 1152), (1, 384), 0), out=buf141)
    del arg69_1
    buf142 = reinterpret_tensor(buf140, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf140  # reuse
    buf143 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_34(c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf142, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf143, (96, 32, 196), (6272, 196, 1), 0), out=buf144)
    buf145 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf146 = reinterpret_tensor(buf144, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf144  # reuse
    buf147 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf148 = buf146; del buf146  # reuse
    buf149 = reinterpret_tensor(buf143, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf143  # reuse
    cpp_fused__softmax_clone_mul_35(c_void_p(buf148.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()))
    buf150 = reinterpret_tensor(buf142, (96, 196, 32), (6272, 32, 1), 0); del buf142  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf148, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf149, (96, 196, 32), (6272, 32, 1), 0), out=buf150)
    buf151 = reinterpret_tensor(buf149, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf149  # reuse
    cpp_fused_clone_36(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()))
    buf152 = reinterpret_tensor(buf150, (1568, 384), (384, 1), 0); del buf150  # reuse
    # Source Nodes: [x_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg71_1, reinterpret_tensor(buf151, (1568, 384), (384, 1), 0), reinterpret_tensor(arg70_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf152)
    del arg70_1
    del arg71_1
    buf153 = buf138; del buf138  # reuse
    buf154 = buf137; del buf137  # reuse
    buf156 = reinterpret_tensor(buf151, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf151  # reuse
    cpp_fused_add_native_layer_norm_37(c_void_p(buf136.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg72_1
    del arg73_1
    buf157 = buf141; del buf141  # reuse
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg75_1, reinterpret_tensor(buf156, (1568, 384), (384, 1), 0), reinterpret_tensor(arg74_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf157)
    del arg74_1
    del arg75_1
    buf158 = reinterpret_tensor(buf157, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf157  # reuse
    cpp_fused_gelu_38(c_void_p(buf158.data_ptr()))
    buf159 = reinterpret_tensor(buf156, (1568, 384), (384, 1), 0); del buf156  # reuse
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg77_1, reinterpret_tensor(buf158, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg76_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf159)
    del arg76_1
    del arg77_1
    buf160 = buf154; del buf154  # reuse
    buf161 = buf153; del buf153  # reuse
    buf163 = empty((8, 14, 14, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_39(c_void_p(buf136.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg78_1
    del arg79_1
    buf164 = reinterpret_tensor(buf158, (1568, 1152), (1152, 1), 0); del buf158  # reuse
    # Source Nodes: [getattr_l__mod___network_2___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf163, (1568, 384), (384, 1), 0), reinterpret_tensor(arg80_1, (384, 1152), (1, 384), 0), out=buf164)
    del arg80_1
    buf165 = reinterpret_tensor(buf163, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf163  # reuse
    buf166 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_40(c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf148, (96, 196, 196), (38416, 196, 1), 0); del buf148  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf165, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf166, (96, 32, 196), (6272, 196, 1), 0), out=buf167)
    buf168 = buf147; del buf147  # reuse
    buf169 = reinterpret_tensor(buf167, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf167  # reuse
    buf170 = buf145; del buf145  # reuse
    buf171 = buf169; del buf169  # reuse
    buf172 = reinterpret_tensor(buf166, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf166  # reuse
    cpp_fused__softmax_clone_mul_41(c_void_p(buf171.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()))
    buf173 = reinterpret_tensor(buf165, (96, 196, 32), (6272, 32, 1), 0); del buf165  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf171, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf172, (96, 196, 32), (6272, 32, 1), 0), out=buf173)
    buf174 = reinterpret_tensor(buf172, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf172  # reuse
    cpp_fused_clone_42(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = reinterpret_tensor(buf173, (1568, 384), (384, 1), 0); del buf173  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg82_1, reinterpret_tensor(buf174, (1568, 384), (384, 1), 0), reinterpret_tensor(arg81_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf175)
    del arg81_1
    del arg82_1
    buf176 = reinterpret_tensor(buf175, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf175  # reuse
    buf177 = buf161; del buf161  # reuse
    buf178 = buf160; del buf160  # reuse
    buf180 = reinterpret_tensor(buf174, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf174  # reuse
    cpp_fused_add_native_layer_norm_43(c_void_p(buf176.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del arg0_1
    del arg83_1
    del arg84_1
    buf181 = buf164; del buf164  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg86_1, reinterpret_tensor(buf180, (1568, 384), (384, 1), 0), reinterpret_tensor(arg85_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf181)
    del arg85_1
    del arg86_1
    buf182 = reinterpret_tensor(buf181, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf181  # reuse
    cpp_fused_gelu_44(c_void_p(buf182.data_ptr()))
    buf183 = reinterpret_tensor(buf180, (1568, 384), (384, 1), 0); del buf180  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg88_1, reinterpret_tensor(buf182, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg87_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf183)
    del arg87_1
    del arg88_1
    buf184 = buf178; del buf178  # reuse
    buf185 = buf177; del buf177  # reuse
    buf187 = reinterpret_tensor(buf159, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf159  # reuse
    cpp_fused_add_native_layer_norm_45(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg89_1
    del arg90_1
    buf188 = reinterpret_tensor(buf182, (1568, 1152), (1152, 1), 0); del buf182  # reuse
    # Source Nodes: [getattr_l__mod___network_2___2___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf187, (1568, 384), (384, 1), 0), reinterpret_tensor(arg91_1, (384, 1152), (1, 384), 0), out=buf188)
    del arg91_1
    buf189 = reinterpret_tensor(buf187, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf187  # reuse
    buf190 = reinterpret_tensor(buf152, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf152  # reuse
    cpp_fused_clone_46(c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf171, (96, 196, 196), (38416, 196, 1), 0); del buf171  # reuse
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf189, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf190, (96, 32, 196), (6272, 196, 1), 0), out=buf191)
    buf192 = buf170; del buf170  # reuse
    buf193 = reinterpret_tensor(buf191, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf191  # reuse
    buf194 = buf168; del buf168  # reuse
    buf195 = buf193; del buf193  # reuse
    buf196 = reinterpret_tensor(buf190, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf190  # reuse
    cpp_fused__softmax_clone_mul_47(c_void_p(buf195.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()))
    buf197 = reinterpret_tensor(buf189, (96, 196, 32), (6272, 32, 1), 0); del buf189  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf196, (96, 196, 32), (6272, 32, 1), 0), out=buf197)
    buf198 = reinterpret_tensor(buf196, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf196  # reuse
    cpp_fused_clone_48(c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    buf199 = reinterpret_tensor(buf197, (1568, 384), (384, 1), 0); del buf197  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg93_1, reinterpret_tensor(buf198, (1568, 384), (384, 1), 0), reinterpret_tensor(arg92_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf199)
    del arg92_1
    del arg93_1
    buf200 = buf185; del buf185  # reuse
    buf201 = buf184; del buf184  # reuse
    buf203 = reinterpret_tensor(buf198, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf198  # reuse
    cpp_fused_add_native_layer_norm_49(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf203.data_ptr()))
    del arg94_1
    del arg95_1
    buf204 = buf188; del buf188  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg97_1, reinterpret_tensor(buf203, (1568, 384), (384, 1), 0), reinterpret_tensor(arg96_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf204)
    del arg96_1
    del arg97_1
    buf205 = reinterpret_tensor(buf204, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf204  # reuse
    cpp_fused_gelu_50(c_void_p(buf205.data_ptr()))
    buf206 = reinterpret_tensor(buf203, (1568, 384), (384, 1), 0); del buf203  # reuse
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg99_1, reinterpret_tensor(buf205, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg98_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf206)
    del arg98_1
    del arg99_1
    buf207 = buf201; del buf201  # reuse
    buf208 = buf200; del buf200  # reuse
    buf210 = reinterpret_tensor(buf136, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf136  # reuse
    cpp_fused_add_native_layer_norm_51(c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg100_1
    del arg101_1
    buf211 = reinterpret_tensor(buf205, (1568, 1152), (1152, 1), 0); del buf205  # reuse
    # Source Nodes: [getattr_l__mod___network_2___3___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf210, (1568, 384), (384, 1), 0), reinterpret_tensor(arg102_1, (384, 1152), (1, 384), 0), out=buf211)
    del arg102_1
    buf212 = reinterpret_tensor(buf210, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf210  # reuse
    buf213 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_52(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    buf214 = reinterpret_tensor(buf195, (96, 196, 196), (38416, 196, 1), 0); del buf195  # reuse
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf212, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf213, (96, 32, 196), (6272, 196, 1), 0), out=buf214)
    buf215 = buf194; del buf194  # reuse
    buf216 = reinterpret_tensor(buf214, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf214  # reuse
    buf217 = buf192; del buf192  # reuse
    buf218 = buf216; del buf216  # reuse
    buf219 = reinterpret_tensor(buf213, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf213  # reuse
    cpp_fused__softmax_clone_mul_53(c_void_p(buf218.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()))
    buf220 = reinterpret_tensor(buf212, (96, 196, 32), (6272, 32, 1), 0); del buf212  # reuse
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf218, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf219, (96, 196, 32), (6272, 32, 1), 0), out=buf220)
    buf221 = reinterpret_tensor(buf219, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf219  # reuse
    cpp_fused_clone_54(c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = reinterpret_tensor(buf220, (1568, 384), (384, 1), 0); del buf220  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg104_1, reinterpret_tensor(buf221, (1568, 384), (384, 1), 0), reinterpret_tensor(arg103_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf222)
    del arg103_1
    del arg104_1
    buf223 = reinterpret_tensor(buf222, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf222  # reuse
    buf224 = buf208; del buf208  # reuse
    buf225 = buf207; del buf207  # reuse
    buf227 = reinterpret_tensor(buf221, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf221  # reuse
    cpp_fused_add_native_layer_norm_55(c_void_p(buf223.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(buf206.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg105_1
    del arg106_1
    buf228 = buf211; del buf211  # reuse
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg108_1, reinterpret_tensor(buf227, (1568, 384), (384, 1), 0), reinterpret_tensor(arg107_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf228)
    del arg107_1
    del arg108_1
    buf229 = reinterpret_tensor(buf228, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf228  # reuse
    cpp_fused_gelu_56(c_void_p(buf229.data_ptr()))
    buf230 = reinterpret_tensor(buf227, (1568, 384), (384, 1), 0); del buf227  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg110_1, reinterpret_tensor(buf229, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg109_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf230)
    del arg109_1
    del arg110_1
    buf231 = buf225; del buf225  # reuse
    buf232 = buf224; del buf224  # reuse
    buf234 = reinterpret_tensor(buf206, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf206  # reuse
    cpp_fused_add_native_layer_norm_57(c_void_p(buf223.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg111_1
    del arg112_1
    buf235 = reinterpret_tensor(buf229, (1568, 1152), (1152, 1), 0); del buf229  # reuse
    # Source Nodes: [getattr_l__mod___network_3___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf234, (1568, 384), (384, 1), 0), reinterpret_tensor(arg113_1, (384, 1152), (1, 384), 0), out=buf235)
    del arg113_1
    buf236 = reinterpret_tensor(buf234, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf234  # reuse
    buf237 = reinterpret_tensor(buf199, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf199  # reuse
    cpp_fused_clone_58(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf218, (96, 196, 196), (38416, 196, 1), 0); del buf218  # reuse
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf236, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf237, (96, 32, 196), (6272, 196, 1), 0), out=buf238)
    buf239 = buf217; del buf217  # reuse
    buf240 = reinterpret_tensor(buf238, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf238  # reuse
    buf241 = buf215; del buf215  # reuse
    buf242 = buf240; del buf240  # reuse
    buf243 = reinterpret_tensor(buf237, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf237  # reuse
    cpp_fused__softmax_clone_mul_59(c_void_p(buf242.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()))
    buf244 = reinterpret_tensor(buf236, (96, 196, 32), (6272, 32, 1), 0); del buf236  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf242, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf243, (96, 196, 32), (6272, 32, 1), 0), out=buf244)
    buf245 = reinterpret_tensor(buf243, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf243  # reuse
    cpp_fused_clone_60(c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    buf246 = reinterpret_tensor(buf244, (1568, 384), (384, 1), 0); del buf244  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg115_1, reinterpret_tensor(buf245, (1568, 384), (384, 1), 0), reinterpret_tensor(arg114_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf246)
    del arg114_1
    del arg115_1
    buf247 = buf232; del buf232  # reuse
    buf248 = buf231; del buf231  # reuse
    buf250 = reinterpret_tensor(buf245, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf245  # reuse
    cpp_fused_add_native_layer_norm_61(c_void_p(buf223.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf250.data_ptr()))
    del arg116_1
    del arg117_1
    buf251 = buf235; del buf235  # reuse
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg119_1, reinterpret_tensor(buf250, (1568, 384), (384, 1), 0), reinterpret_tensor(arg118_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf251)
    del arg118_1
    del arg119_1
    buf252 = reinterpret_tensor(buf251, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf251  # reuse
    cpp_fused_gelu_62(c_void_p(buf252.data_ptr()))
    buf253 = reinterpret_tensor(buf250, (1568, 384), (384, 1), 0); del buf250  # reuse
    # Source Nodes: [x_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg121_1, reinterpret_tensor(buf252, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg120_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf253)
    del arg120_1
    del arg121_1
    buf254 = buf248; del buf248  # reuse
    buf255 = buf247; del buf247  # reuse
    buf257 = reinterpret_tensor(buf183, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf183  # reuse
    cpp_fused_add_native_layer_norm_63(c_void_p(buf223.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()))
    del arg122_1
    del arg123_1
    buf258 = reinterpret_tensor(buf252, (1568, 1152), (1152, 1), 0); del buf252  # reuse
    # Source Nodes: [getattr_l__mod___network_3___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf257, (1568, 384), (384, 1), 0), reinterpret_tensor(arg124_1, (384, 1152), (1, 384), 0), out=buf258)
    del arg124_1
    buf259 = reinterpret_tensor(buf257, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf257  # reuse
    buf260 = reinterpret_tensor(buf176, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf176  # reuse
    cpp_fused_clone_64(c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = reinterpret_tensor(buf242, (96, 196, 196), (38416, 196, 1), 0); del buf242  # reuse
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf260, (96, 32, 196), (6272, 196, 1), 0), out=buf261)
    buf262 = buf241; del buf241  # reuse
    buf263 = reinterpret_tensor(buf261, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf261  # reuse
    buf264 = buf239; del buf239  # reuse
    buf265 = buf263; del buf263  # reuse
    buf266 = reinterpret_tensor(buf260, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf260  # reuse
    cpp_fused__softmax_clone_mul_65(c_void_p(buf265.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = reinterpret_tensor(buf259, (96, 196, 32), (6272, 32, 1), 0); del buf259  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf266, (96, 196, 32), (6272, 32, 1), 0), out=buf267)
    buf268 = reinterpret_tensor(buf266, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf266  # reuse
    cpp_fused_clone_66(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (1568, 384), (384, 1), 0); del buf267  # reuse
    # Source Nodes: [x_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg126_1, reinterpret_tensor(buf268, (1568, 384), (384, 1), 0), reinterpret_tensor(arg125_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf269)
    del arg125_1
    del arg126_1
    buf270 = reinterpret_tensor(buf269, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf269  # reuse
    buf271 = buf255; del buf255  # reuse
    buf272 = buf254; del buf254  # reuse
    buf274 = reinterpret_tensor(buf268, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf268  # reuse
    cpp_fused_add_native_layer_norm_67(c_void_p(buf270.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(buf274.data_ptr()))
    del arg127_1
    del arg128_1
    buf275 = buf258; del buf258  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg130_1, reinterpret_tensor(buf274, (1568, 384), (384, 1), 0), reinterpret_tensor(arg129_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf275)
    del arg129_1
    del arg130_1
    buf276 = reinterpret_tensor(buf275, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf275  # reuse
    cpp_fused_gelu_68(c_void_p(buf276.data_ptr()))
    buf277 = reinterpret_tensor(buf274, (1568, 384), (384, 1), 0); del buf274  # reuse
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg132_1, reinterpret_tensor(buf276, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg131_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf277)
    del arg131_1
    del arg132_1
    buf278 = buf272; del buf272  # reuse
    buf279 = buf271; del buf271  # reuse
    buf281 = reinterpret_tensor(buf253, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf253  # reuse
    cpp_fused_add_native_layer_norm_69(c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()))
    del arg133_1
    del arg134_1
    buf282 = reinterpret_tensor(buf276, (1568, 1152), (1152, 1), 0); del buf276  # reuse
    # Source Nodes: [getattr_l__mod___network_3___2___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf281, (1568, 384), (384, 1), 0), reinterpret_tensor(arg135_1, (384, 1152), (1, 384), 0), out=buf282)
    del arg135_1
    buf283 = reinterpret_tensor(buf281, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf281  # reuse
    buf284 = reinterpret_tensor(buf246, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf246  # reuse
    cpp_fused_clone_70(c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    buf285 = reinterpret_tensor(buf265, (96, 196, 196), (38416, 196, 1), 0); del buf265  # reuse
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf283, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf284, (96, 32, 196), (6272, 196, 1), 0), out=buf285)
    buf286 = buf264; del buf264  # reuse
    buf287 = reinterpret_tensor(buf285, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf285  # reuse
    buf288 = buf262; del buf262  # reuse
    buf289 = buf287; del buf287  # reuse
    buf290 = reinterpret_tensor(buf284, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf284  # reuse
    cpp_fused__softmax_clone_mul_71(c_void_p(buf289.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()))
    buf291 = reinterpret_tensor(buf283, (96, 196, 32), (6272, 32, 1), 0); del buf283  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf289, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf290, (96, 196, 32), (6272, 32, 1), 0), out=buf291)
    buf292 = reinterpret_tensor(buf290, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf290  # reuse
    cpp_fused_clone_72(c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()))
    buf293 = reinterpret_tensor(buf291, (1568, 384), (384, 1), 0); del buf291  # reuse
    # Source Nodes: [x_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg137_1, reinterpret_tensor(buf292, (1568, 384), (384, 1), 0), reinterpret_tensor(arg136_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf293)
    del arg136_1
    del arg137_1
    buf294 = buf279; del buf279  # reuse
    buf295 = buf278; del buf278  # reuse
    buf297 = reinterpret_tensor(buf292, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf292  # reuse
    cpp_fused_add_native_layer_norm_73(c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg138_1
    del arg139_1
    buf298 = buf282; del buf282  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg141_1, reinterpret_tensor(buf297, (1568, 384), (384, 1), 0), reinterpret_tensor(arg140_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf298)
    del arg140_1
    del arg141_1
    buf299 = reinterpret_tensor(buf298, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf298  # reuse
    cpp_fused_gelu_74(c_void_p(buf299.data_ptr()))
    buf300 = reinterpret_tensor(buf297, (1568, 384), (384, 1), 0); del buf297  # reuse
    # Source Nodes: [x_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg143_1, reinterpret_tensor(buf299, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg142_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf300)
    del arg142_1
    del arg143_1
    buf301 = buf295; del buf295  # reuse
    buf302 = buf294; del buf294  # reuse
    buf304 = reinterpret_tensor(buf230, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf230  # reuse
    cpp_fused_add_native_layer_norm_75(c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()))
    del arg144_1
    del arg145_1
    buf305 = reinterpret_tensor(buf299, (1568, 1152), (1152, 1), 0); del buf299  # reuse
    # Source Nodes: [getattr_l__mod___network_3___3___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf304, (1568, 384), (384, 1), 0), reinterpret_tensor(arg146_1, (384, 1152), (1, 384), 0), out=buf305)
    del arg146_1
    buf306 = reinterpret_tensor(buf304, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf304  # reuse
    buf307 = reinterpret_tensor(buf223, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf223  # reuse
    cpp_fused_clone_76(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf289, (96, 196, 196), (38416, 196, 1), 0); del buf289  # reuse
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf306, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf307, (96, 32, 196), (6272, 196, 1), 0), out=buf308)
    buf309 = buf288; del buf288  # reuse
    buf310 = reinterpret_tensor(buf308, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf308  # reuse
    buf311 = buf286; del buf286  # reuse
    buf312 = buf310; del buf310  # reuse
    buf313 = reinterpret_tensor(buf307, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf307  # reuse
    cpp_fused__softmax_clone_mul_77(c_void_p(buf312.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()))
    buf314 = reinterpret_tensor(buf306, (96, 196, 32), (6272, 32, 1), 0); del buf306  # reuse
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf312, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf313, (96, 196, 32), (6272, 32, 1), 0), out=buf314)
    buf315 = reinterpret_tensor(buf313, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf313  # reuse
    cpp_fused_clone_78(c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()))
    buf316 = reinterpret_tensor(buf314, (1568, 384), (384, 1), 0); del buf314  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg148_1, reinterpret_tensor(buf315, (1568, 384), (384, 1), 0), reinterpret_tensor(arg147_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf316)
    del arg147_1
    del arg148_1
    buf317 = reinterpret_tensor(buf316, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf316  # reuse
    buf318 = buf302; del buf302  # reuse
    buf319 = buf301; del buf301  # reuse
    buf321 = reinterpret_tensor(buf315, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf315  # reuse
    cpp_fused_add_native_layer_norm_79(c_void_p(buf317.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf321.data_ptr()))
    del arg149_1
    del arg150_1
    buf322 = buf305; del buf305  # reuse
    # Source Nodes: [x_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg152_1, reinterpret_tensor(buf321, (1568, 384), (384, 1), 0), reinterpret_tensor(arg151_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf322)
    del arg151_1
    del arg152_1
    buf323 = reinterpret_tensor(buf322, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf322  # reuse
    cpp_fused_gelu_80(c_void_p(buf323.data_ptr()))
    buf324 = reinterpret_tensor(buf321, (1568, 384), (384, 1), 0); del buf321  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg154_1, reinterpret_tensor(buf323, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg153_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf324)
    del arg153_1
    del arg154_1
    buf325 = buf319; del buf319  # reuse
    buf326 = buf318; del buf318  # reuse
    buf328 = reinterpret_tensor(buf300, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf300  # reuse
    cpp_fused_add_native_layer_norm_81(c_void_p(buf317.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(buf328.data_ptr()))
    del arg155_1
    del arg156_1
    buf329 = reinterpret_tensor(buf323, (1568, 1152), (1152, 1), 0); del buf323  # reuse
    # Source Nodes: [getattr_l__mod___network_3___4___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf328, (1568, 384), (384, 1), 0), reinterpret_tensor(arg157_1, (384, 1152), (1, 384), 0), out=buf329)
    del arg157_1
    buf330 = reinterpret_tensor(buf328, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf328  # reuse
    buf331 = reinterpret_tensor(buf293, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf293  # reuse
    cpp_fused_clone_82(c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf331.data_ptr()))
    buf332 = reinterpret_tensor(buf312, (96, 196, 196), (38416, 196, 1), 0); del buf312  # reuse
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf330, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf331, (96, 32, 196), (6272, 196, 1), 0), out=buf332)
    buf333 = buf311; del buf311  # reuse
    buf334 = reinterpret_tensor(buf332, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf332  # reuse
    buf335 = buf309; del buf309  # reuse
    buf336 = buf334; del buf334  # reuse
    buf337 = reinterpret_tensor(buf331, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf331  # reuse
    cpp_fused__softmax_clone_mul_83(c_void_p(buf336.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()))
    buf338 = reinterpret_tensor(buf330, (96, 196, 32), (6272, 32, 1), 0); del buf330  # reuse
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf336, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf337, (96, 196, 32), (6272, 32, 1), 0), out=buf338)
    buf339 = reinterpret_tensor(buf337, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf337  # reuse
    cpp_fused_clone_84(c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()))
    buf340 = reinterpret_tensor(buf338, (1568, 384), (384, 1), 0); del buf338  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg159_1, reinterpret_tensor(buf339, (1568, 384), (384, 1), 0), reinterpret_tensor(arg158_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf340)
    del arg158_1
    del arg159_1
    buf341 = buf326; del buf326  # reuse
    buf342 = buf325; del buf325  # reuse
    buf344 = reinterpret_tensor(buf339, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf339  # reuse
    cpp_fused_add_native_layer_norm_85(c_void_p(buf317.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf344.data_ptr()))
    del arg160_1
    del arg161_1
    buf345 = buf329; del buf329  # reuse
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg163_1, reinterpret_tensor(buf344, (1568, 384), (384, 1), 0), reinterpret_tensor(arg162_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf345)
    del arg162_1
    del arg163_1
    buf346 = reinterpret_tensor(buf345, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf345  # reuse
    cpp_fused_gelu_86(c_void_p(buf346.data_ptr()))
    buf347 = reinterpret_tensor(buf344, (1568, 384), (384, 1), 0); del buf344  # reuse
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg165_1, reinterpret_tensor(buf346, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg164_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf347)
    del arg164_1
    del arg165_1
    buf348 = buf342; del buf342  # reuse
    buf349 = buf341; del buf341  # reuse
    buf351 = reinterpret_tensor(buf277, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf277  # reuse
    cpp_fused_add_native_layer_norm_87(c_void_p(buf317.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf351.data_ptr()))
    del arg166_1
    del arg167_1
    buf352 = reinterpret_tensor(buf346, (1568, 1152), (1152, 1), 0); del buf346  # reuse
    # Source Nodes: [getattr_l__mod___network_3___5___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf351, (1568, 384), (384, 1), 0), reinterpret_tensor(arg168_1, (384, 1152), (1, 384), 0), out=buf352)
    del arg168_1
    buf353 = reinterpret_tensor(buf351, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf351  # reuse
    buf354 = reinterpret_tensor(buf270, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf270  # reuse
    cpp_fused_clone_88(c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    buf355 = reinterpret_tensor(buf336, (96, 196, 196), (38416, 196, 1), 0); del buf336  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf353, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf354, (96, 32, 196), (6272, 196, 1), 0), out=buf355)
    buf356 = buf335; del buf335  # reuse
    buf357 = reinterpret_tensor(buf355, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf355  # reuse
    buf358 = buf333; del buf333  # reuse
    buf359 = buf357; del buf357  # reuse
    buf360 = reinterpret_tensor(buf354, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf354  # reuse
    cpp_fused__softmax_clone_mul_89(c_void_p(buf359.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()))
    buf361 = reinterpret_tensor(buf353, (96, 196, 32), (6272, 32, 1), 0); del buf353  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf359, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf360, (96, 196, 32), (6272, 32, 1), 0), out=buf361)
    buf362 = reinterpret_tensor(buf360, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf360  # reuse
    cpp_fused_clone_90(c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = reinterpret_tensor(buf361, (1568, 384), (384, 1), 0); del buf361  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg170_1, reinterpret_tensor(buf362, (1568, 384), (384, 1), 0), reinterpret_tensor(arg169_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf363)
    del arg169_1
    del arg170_1
    buf364 = reinterpret_tensor(buf363, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf363  # reuse
    buf365 = buf349; del buf349  # reuse
    buf366 = buf348; del buf348  # reuse
    buf368 = reinterpret_tensor(buf362, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf362  # reuse
    cpp_fused_add_native_layer_norm_91(c_void_p(buf364.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf368.data_ptr()))
    del arg171_1
    del arg172_1
    buf369 = buf352; del buf352  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg174_1, reinterpret_tensor(buf368, (1568, 384), (384, 1), 0), reinterpret_tensor(arg173_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf369)
    del arg173_1
    del arg174_1
    buf370 = reinterpret_tensor(buf369, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf369  # reuse
    cpp_fused_gelu_92(c_void_p(buf370.data_ptr()))
    buf371 = reinterpret_tensor(buf368, (1568, 384), (384, 1), 0); del buf368  # reuse
    # Source Nodes: [x_166], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg176_1, reinterpret_tensor(buf370, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg175_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf371)
    del arg175_1
    del arg176_1
    buf372 = buf366; del buf366  # reuse
    buf373 = buf365; del buf365  # reuse
    buf375 = reinterpret_tensor(buf347, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf347  # reuse
    cpp_fused_add_native_layer_norm_93(c_void_p(buf364.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()))
    del arg177_1
    del arg178_1
    buf376 = reinterpret_tensor(buf370, (1568, 1152), (1152, 1), 0); del buf370  # reuse
    # Source Nodes: [getattr_l__mod___network_3___6___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf375, (1568, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 1152), (1, 384), 0), out=buf376)
    del arg179_1
    buf377 = reinterpret_tensor(buf375, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf375  # reuse
    buf378 = reinterpret_tensor(buf340, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf340  # reuse
    cpp_fused_clone_94(c_void_p(buf376.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = reinterpret_tensor(buf359, (96, 196, 196), (38416, 196, 1), 0); del buf359  # reuse
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf377, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf378, (96, 32, 196), (6272, 196, 1), 0), out=buf379)
    buf380 = buf358; del buf358  # reuse
    buf381 = reinterpret_tensor(buf379, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf379  # reuse
    buf382 = buf356; del buf356  # reuse
    buf383 = buf381; del buf381  # reuse
    buf384 = reinterpret_tensor(buf378, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf378  # reuse
    cpp_fused__softmax_clone_mul_95(c_void_p(buf383.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf384.data_ptr()))
    buf385 = reinterpret_tensor(buf377, (96, 196, 32), (6272, 32, 1), 0); del buf377  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf383, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf384, (96, 196, 32), (6272, 32, 1), 0), out=buf385)
    buf386 = reinterpret_tensor(buf384, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf384  # reuse
    cpp_fused_clone_96(c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()))
    buf387 = reinterpret_tensor(buf385, (1568, 384), (384, 1), 0); del buf385  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg181_1, reinterpret_tensor(buf386, (1568, 384), (384, 1), 0), reinterpret_tensor(arg180_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf387)
    del arg180_1
    del arg181_1
    buf388 = buf373; del buf373  # reuse
    buf389 = buf372; del buf372  # reuse
    buf391 = reinterpret_tensor(buf386, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf386  # reuse
    cpp_fused_add_native_layer_norm_97(c_void_p(buf364.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()))
    del arg182_1
    del arg183_1
    buf392 = buf376; del buf376  # reuse
    # Source Nodes: [x_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg185_1, reinterpret_tensor(buf391, (1568, 384), (384, 1), 0), reinterpret_tensor(arg184_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf392)
    del arg184_1
    del arg185_1
    buf393 = reinterpret_tensor(buf392, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf392  # reuse
    cpp_fused_gelu_98(c_void_p(buf393.data_ptr()))
    buf394 = reinterpret_tensor(buf391, (1568, 384), (384, 1), 0); del buf391  # reuse
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg187_1, reinterpret_tensor(buf393, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg186_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf394)
    del arg186_1
    del arg187_1
    buf395 = buf389; del buf389  # reuse
    buf396 = buf388; del buf388  # reuse
    buf398 = reinterpret_tensor(buf324, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf324  # reuse
    cpp_fused_add_native_layer_norm_99(c_void_p(buf364.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf398.data_ptr()))
    del arg188_1
    del arg189_1
    buf399 = reinterpret_tensor(buf393, (1568, 1152), (1152, 1), 0); del buf393  # reuse
    # Source Nodes: [getattr_l__mod___network_3___7___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf398, (1568, 384), (384, 1), 0), reinterpret_tensor(arg190_1, (384, 1152), (1, 384), 0), out=buf399)
    del arg190_1
    buf400 = reinterpret_tensor(buf398, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf398  # reuse
    buf401 = reinterpret_tensor(buf317, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf317  # reuse
    cpp_fused_clone_100(c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(buf401.data_ptr()))
    buf402 = reinterpret_tensor(buf383, (96, 196, 196), (38416, 196, 1), 0); del buf383  # reuse
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf400, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf401, (96, 32, 196), (6272, 196, 1), 0), out=buf402)
    buf403 = buf382; del buf382  # reuse
    buf404 = reinterpret_tensor(buf402, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf402  # reuse
    buf405 = buf380; del buf380  # reuse
    buf406 = buf404; del buf404  # reuse
    buf407 = reinterpret_tensor(buf401, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf401  # reuse
    cpp_fused__softmax_clone_mul_101(c_void_p(buf406.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf407.data_ptr()))
    buf408 = reinterpret_tensor(buf400, (96, 196, 32), (6272, 32, 1), 0); del buf400  # reuse
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf406, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf407, (96, 196, 32), (6272, 32, 1), 0), out=buf408)
    buf409 = reinterpret_tensor(buf407, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf407  # reuse
    cpp_fused_clone_102(c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = reinterpret_tensor(buf408, (1568, 384), (384, 1), 0); del buf408  # reuse
    # Source Nodes: [x_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg192_1, reinterpret_tensor(buf409, (1568, 384), (384, 1), 0), reinterpret_tensor(arg191_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf410)
    del arg191_1
    del arg192_1
    buf411 = reinterpret_tensor(buf410, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf410  # reuse
    buf412 = buf396; del buf396  # reuse
    buf413 = buf395; del buf395  # reuse
    buf415 = reinterpret_tensor(buf409, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf409  # reuse
    cpp_fused_add_native_layer_norm_103(c_void_p(buf411.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()))
    del arg193_1
    del arg194_1
    buf416 = buf399; del buf399  # reuse
    # Source Nodes: [x_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg196_1, reinterpret_tensor(buf415, (1568, 384), (384, 1), 0), reinterpret_tensor(arg195_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf416)
    del arg195_1
    del arg196_1
    buf417 = reinterpret_tensor(buf416, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf416  # reuse
    cpp_fused_gelu_104(c_void_p(buf417.data_ptr()))
    buf418 = reinterpret_tensor(buf415, (1568, 384), (384, 1), 0); del buf415  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg198_1, reinterpret_tensor(buf417, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg197_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf418)
    del arg197_1
    del arg198_1
    buf419 = buf413; del buf413  # reuse
    buf420 = buf412; del buf412  # reuse
    buf422 = reinterpret_tensor(buf394, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf394  # reuse
    cpp_fused_add_native_layer_norm_105(c_void_p(buf411.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf422.data_ptr()))
    del arg199_1
    del arg200_1
    buf423 = reinterpret_tensor(buf417, (1568, 1152), (1152, 1), 0); del buf417  # reuse
    # Source Nodes: [getattr_l__mod___network_4___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf422, (1568, 384), (384, 1), 0), reinterpret_tensor(arg201_1, (384, 1152), (1, 384), 0), out=buf423)
    del arg201_1
    buf424 = reinterpret_tensor(buf422, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf422  # reuse
    buf425 = reinterpret_tensor(buf387, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf387  # reuse
    cpp_fused_clone_106(c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()))
    buf426 = reinterpret_tensor(buf406, (96, 196, 196), (38416, 196, 1), 0); del buf406  # reuse
    # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf424, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf425, (96, 32, 196), (6272, 196, 1), 0), out=buf426)
    buf427 = buf405; del buf405  # reuse
    buf428 = reinterpret_tensor(buf426, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf426  # reuse
    buf429 = buf403; del buf403  # reuse
    buf430 = buf428; del buf428  # reuse
    buf431 = reinterpret_tensor(buf425, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf425  # reuse
    cpp_fused__softmax_clone_mul_107(c_void_p(buf430.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf431.data_ptr()))
    buf432 = reinterpret_tensor(buf424, (96, 196, 32), (6272, 32, 1), 0); del buf424  # reuse
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf430, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf431, (96, 196, 32), (6272, 32, 1), 0), out=buf432)
    buf433 = reinterpret_tensor(buf431, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf431  # reuse
    cpp_fused_clone_108(c_void_p(buf432.data_ptr()), c_void_p(buf433.data_ptr()))
    buf434 = reinterpret_tensor(buf432, (1568, 384), (384, 1), 0); del buf432  # reuse
    # Source Nodes: [x_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg203_1, reinterpret_tensor(buf433, (1568, 384), (384, 1), 0), reinterpret_tensor(arg202_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf434)
    del arg202_1
    del arg203_1
    buf435 = buf420; del buf420  # reuse
    buf436 = buf419; del buf419  # reuse
    buf438 = reinterpret_tensor(buf433, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf433  # reuse
    cpp_fused_add_native_layer_norm_109(c_void_p(buf411.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf438.data_ptr()))
    del arg204_1
    del arg205_1
    buf439 = buf423; del buf423  # reuse
    # Source Nodes: [x_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg207_1, reinterpret_tensor(buf438, (1568, 384), (384, 1), 0), reinterpret_tensor(arg206_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf439)
    del arg206_1
    del arg207_1
    buf440 = reinterpret_tensor(buf439, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf439  # reuse
    cpp_fused_gelu_110(c_void_p(buf440.data_ptr()))
    buf441 = reinterpret_tensor(buf438, (1568, 384), (384, 1), 0); del buf438  # reuse
    # Source Nodes: [x_200], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg209_1, reinterpret_tensor(buf440, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg208_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf441)
    del arg208_1
    del arg209_1
    buf442 = buf436; del buf436  # reuse
    buf443 = buf435; del buf435  # reuse
    buf445 = reinterpret_tensor(buf371, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf371  # reuse
    cpp_fused_add_native_layer_norm_111(c_void_p(buf411.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf445.data_ptr()))
    del arg210_1
    del arg211_1
    buf446 = reinterpret_tensor(buf440, (1568, 1152), (1152, 1), 0); del buf440  # reuse
    # Source Nodes: [getattr_l__mod___network_4___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf445, (1568, 384), (384, 1), 0), reinterpret_tensor(arg212_1, (384, 1152), (1, 384), 0), out=buf446)
    del arg212_1
    buf447 = reinterpret_tensor(buf445, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf445  # reuse
    buf448 = reinterpret_tensor(buf364, (8, 12, 32, 196), (75264, 6272, 196, 1), 0); del buf364  # reuse
    cpp_fused_clone_112(c_void_p(buf446.data_ptr()), c_void_p(buf447.data_ptr()), c_void_p(buf448.data_ptr()))
    buf449 = reinterpret_tensor(buf430, (96, 196, 196), (38416, 196, 1), 0); del buf430  # reuse
    # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf447, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf448, (96, 32, 196), (6272, 196, 1), 0), out=buf449)
    buf450 = buf429; del buf429  # reuse
    buf451 = reinterpret_tensor(buf449, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf449  # reuse
    buf452 = buf427; del buf427  # reuse
    buf453 = buf451; del buf451  # reuse
    buf454 = reinterpret_tensor(buf448, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf448  # reuse
    cpp_fused__softmax_clone_mul_113(c_void_p(buf453.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf454.data_ptr()))
    del buf450
    del buf452
    buf455 = reinterpret_tensor(buf447, (96, 196, 32), (6272, 32, 1), 0); del buf447  # reuse
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf453, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf454, (96, 196, 32), (6272, 32, 1), 0), out=buf455)
    del buf453
    buf456 = reinterpret_tensor(buf454, (8, 196, 12, 32), (75264, 384, 32, 1), 0); del buf454  # reuse
    cpp_fused_clone_114(c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()))
    buf457 = reinterpret_tensor(buf455, (1568, 384), (384, 1), 0); del buf455  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg214_1, reinterpret_tensor(buf456, (1568, 384), (384, 1), 0), reinterpret_tensor(arg213_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf457)
    del arg213_1
    del arg214_1
    buf458 = reinterpret_tensor(buf457, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf457  # reuse
    buf459 = buf443; del buf443  # reuse
    buf460 = buf442; del buf442  # reuse
    buf462 = reinterpret_tensor(buf456, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf456  # reuse
    cpp_fused_add_native_layer_norm_115(c_void_p(buf458.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf462.data_ptr()))
    del arg215_1
    del arg216_1
    del buf411
    del buf418
    del buf434
    del buf441
    del buf459
    del buf460
    buf463 = buf446; del buf446  # reuse
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg218_1, reinterpret_tensor(buf462, (1568, 384), (384, 1), 0), reinterpret_tensor(arg217_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf463)
    del arg217_1
    del arg218_1
    buf464 = reinterpret_tensor(buf463, (8, 14, 14, 1152), (225792, 16128, 1152, 1), 0); del buf463  # reuse
    cpp_fused_gelu_116(c_void_p(buf464.data_ptr()))
    buf465 = reinterpret_tensor(buf462, (1568, 384), (384, 1), 0); del buf462  # reuse
    # Source Nodes: [x_211], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg220_1, reinterpret_tensor(buf464, (1568, 1152), (1152, 1), 0), reinterpret_tensor(arg219_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf465)
    del arg219_1
    del arg220_1
    del buf464
    buf466 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf467 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf469 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_117(c_void_p(arg1_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf469.data_ptr()))
    del arg221_1
    del arg222_1
    buf470 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___post_network_0_attn_kv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (1576, 384), (384, 1), 0), reinterpret_tensor(arg223_1, (384, 768), (1, 384), 0), out=buf470)
    del arg223_1
    buf471 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___post_network_0_attn_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf469, (8, 384), (75648, 1), 0), reinterpret_tensor(arg224_1, (384, 384), (1, 384), 0), out=buf471)
    del arg224_1
    buf472 = reinterpret_tensor(buf471, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf471  # reuse
    buf473 = reinterpret_tensor(buf469, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf469  # reuse
    cpp_fused_clone_mul_118(c_void_p(buf472.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf473.data_ptr()))
    buf474 = empty((96, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_62], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf472, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf473, (96, 32, 197), (6304, 197, 1), 0), out=buf474)
    buf475 = empty_strided((8, 12, 1, 1), (12, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf476 = reinterpret_tensor(buf474, (8, 12, 1, 197), (2364, 197, 18912, 1), 0); del buf474  # reuse
    buf477 = empty_strided((8, 12, 1, 1), (12, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf478 = reinterpret_tensor(buf476, (8, 12, 1, 197), (2364, 197, 197, 1), 0); del buf476  # reuse
    buf479 = reinterpret_tensor(buf473, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf473  # reuse
    cpp_fused__softmax_clone_119(c_void_p(buf478.data_ptr()), c_void_p(buf470.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = reinterpret_tensor(buf472, (96, 1, 32), (32, 32, 1), 0); del buf472  # reuse
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf478, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf479, (96, 197, 32), (6304, 32, 1), 0), out=buf480)
    buf481 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_embed_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg226_1, reinterpret_tensor(buf480, (8, 384), (384, 1), 0), reinterpret_tensor(arg225_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf481)
    del arg225_1
    del arg226_1
    buf482 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf483 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf485 = reinterpret_tensor(buf480, (8, 1, 384), (384, 384, 1), 0); del buf480  # reuse
    cpp_fused_add_native_layer_norm_120(c_void_p(arg1_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()))
    del arg227_1
    del arg228_1
    buf486 = empty((8, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg230_1, reinterpret_tensor(buf485, (8, 384), (384, 1), 0), reinterpret_tensor(arg229_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf486)
    del arg229_1
    del arg230_1
    buf487 = reinterpret_tensor(buf486, (8, 1, 1152), (1152, 1152, 1), 0); del buf486  # reuse
    cpp_fused_gelu_121(c_void_p(buf487.data_ptr()))
    buf488 = reinterpret_tensor(buf485, (8, 384), (384, 1), 0); del buf485  # reuse
    # Source Nodes: [x_222], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg232_1, reinterpret_tensor(buf487, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg231_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf488)
    del arg231_1
    del arg232_1
    buf489 = reinterpret_tensor(buf479, (8, 197, 384), (75648, 1, 197), 0); del buf479  # reuse
    buf490 = buf467; del buf467  # reuse
    buf491 = buf466; del buf466  # reuse
    buf493 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_122(c_void_p(arg1_1.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()))
    del arg1_1
    del arg233_1
    del arg234_1
    del buf458
    buf494 = buf470; del buf470  # reuse
    # Source Nodes: [l__mod___post_network_1_attn_kv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (1576, 384), (384, 1), 0), reinterpret_tensor(arg235_1, (384, 768), (1, 384), 0), out=buf494)
    del arg235_1
    buf495 = buf488; del buf488  # reuse
    # Source Nodes: [l__mod___post_network_1_attn_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf493, (8, 384), (75648, 1), 0), reinterpret_tensor(arg236_1, (384, 384), (1, 384), 0), out=buf495)
    del arg236_1
    buf496 = reinterpret_tensor(buf495, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf495  # reuse
    buf497 = reinterpret_tensor(buf493, (8, 12, 32, 197), (75648, 6304, 197, 1), 0); del buf493  # reuse
    cpp_fused_clone_mul_123(c_void_p(buf496.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf478, (96, 1, 197), (197, 197, 1), 0); del buf478  # reuse
    # Source Nodes: [attn_65], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf496, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf497, (96, 32, 197), (6304, 197, 1), 0), out=buf498)
    buf499 = buf477; del buf477  # reuse
    buf500 = reinterpret_tensor(buf498, (8, 12, 1, 197), (2364, 197, 18912, 1), 0); del buf498  # reuse
    buf501 = buf475; del buf475  # reuse
    buf502 = reinterpret_tensor(buf500, (8, 12, 1, 197), (2364, 197, 197, 1), 0); del buf500  # reuse
    buf503 = reinterpret_tensor(buf497, (8, 12, 197, 32), (75648, 6304, 32, 1), 0); del buf497  # reuse
    cpp_fused__softmax_clone_124(c_void_p(buf502.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()))
    del buf494
    del buf499
    del buf501
    buf504 = reinterpret_tensor(buf496, (96, 1, 32), (32, 32, 1), 0); del buf496  # reuse
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf502, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf503, (96, 197, 32), (6304, 32, 1), 0), out=buf504)
    del buf502
    buf505 = buf481; del buf481  # reuse
    # Source Nodes: [cls_embed_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg238_1, reinterpret_tensor(buf504, (8, 384), (384, 1), 0), reinterpret_tensor(arg237_1, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf505)
    del arg237_1
    del arg238_1
    buf506 = buf483; del buf483  # reuse
    buf507 = buf482; del buf482  # reuse
    buf509 = reinterpret_tensor(buf504, (8, 1, 384), (384, 384, 1), 0); del buf504  # reuse
    cpp_fused_add_native_layer_norm_125(c_void_p(buf489.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf509.data_ptr()))
    del arg239_1
    del arg240_1
    del buf506
    del buf507
    buf510 = reinterpret_tensor(buf487, (8, 1152), (1152, 1), 0); del buf487  # reuse
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg242_1, reinterpret_tensor(buf509, (8, 384), (384, 1), 0), reinterpret_tensor(arg241_1, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf510)
    del arg241_1
    del arg242_1
    buf511 = reinterpret_tensor(buf510, (8, 1, 1152), (1152, 1152, 1), 0); del buf510  # reuse
    cpp_fused_gelu_126(c_void_p(buf511.data_ptr()))
    buf512 = reinterpret_tensor(buf509, (8, 384), (384, 1), 0); del buf509  # reuse
    # Source Nodes: [x_229], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg244_1, reinterpret_tensor(buf511, (8, 1152), (1152, 1), 0), reinterpret_tensor(arg243_1, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf512)
    del arg243_1
    del arg244_1
    del buf511
    buf513 = buf491; del buf491  # reuse
    buf514 = buf490; del buf490  # reuse
    buf516 = reinterpret_tensor(buf503, (8, 197, 384), (75648, 1, 197), 0); del buf503  # reuse
    buf517 = reinterpret_tensor(buf465, (8, 196, 384), (75264, 384, 1), 0); del buf465  # reuse
    cpp_fused_cat_clone_native_layer_norm_127(c_void_p(buf489.data_ptr()), c_void_p(buf505.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()))
    del arg245_1
    del arg246_1
    del buf489
    del buf505
    del buf513
    del buf514
    buf518 = empty((1568, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [aux], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf517, (1568, 384), (384, 1), 0), reinterpret_tensor(arg249_1, (384, 1000), (1, 384), 0), out=buf518)
    del arg249_1
    del buf517
    buf519 = empty((8, 1000), device='cpu', dtype=torch.float32)
    buf521 = buf512; del buf512  # reuse
    cpp_fused_add_addmm_max_128(c_void_p(buf518.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf521.data_ptr()))
    del arg250_1
    del buf516
    del buf518
    buf522 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [out_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg248_1, buf521, reinterpret_tensor(arg247_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf522)
    del arg247_1
    del arg248_1
    del buf521
    buf523 = buf519; del buf519  # reuse
    cpp_fused_add_mul_129(c_void_p(buf523.data_ptr()), c_void_p(buf522.data_ptr()))
    return (buf523, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 14, 14, 384), (75264, 5376, 384, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg254_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg257_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg260_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
