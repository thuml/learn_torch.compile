
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
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
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (49L*x1) + (147L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (147L*x0))] = tmp0;
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
            }
        }
    }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (16L*x1) + (16L*x1_inner) + (1024L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (1024L*x0)), static_cast<long>(64L));
                    }
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
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (4L*x1) + (4L*x1_inner) + (768L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (192L*x2) + (768L*x0)));
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
                        auto tmp0 = in_ptr5[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr5[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                    }
                    tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    tmp7.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(100352.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    auto tmp16 = at::vec::clamp_min(tmp15, decltype(tmp15)(0));
                    tmp16.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_view_4 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>(x3 + (192L*x2) + (192L*x2_inner) + (5376L*x1) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (5376L*x2_inner) + (150528L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (150528L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_im2col_native_layer_norm_permute_view_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       long* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0 + (2L*x1));
                out_ptr0[static_cast<long>(x1 + (14L*x0))] = tmp0;
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (5376L*x3) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr1 + static_cast<long>(x1 + (192L*x3) + (5376L*x2) + (150528L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(192L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(5376L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(5568L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr2 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr2[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
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
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr1[static_cast<long>((14L*(c10::div_floor_integer(x3, 3L))) + (c10::div_floor_integer(x2, 14L)))];
                                auto tmp4 = in_ptr1[static_cast<long>((14L*(static_cast<long>(x3) % static_cast<long>(3L))) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = decltype(tmp0)(tmp0 + 30);
                                auto tmp2 = tmp0 < 0;
                                auto tmp3 = tmp2 ? tmp1 : tmp0;
                                TORCH_CHECK((0 <= tmp3) & (tmp3 < 30L), "index out of bounds: 0 <= tmp3 < 30L")
                                auto tmp5 = decltype(tmp4)(tmp4 + 30);
                                auto tmp6 = tmp4 < 0;
                                auto tmp7 = tmp6 ? tmp5 : tmp4;
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30L), "index out of bounds: 0 <= tmp7 < 30L")
                                auto tmp8 = c10::convert<long>((-1L) + tmp3);
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(28);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = c10::convert<long>((-1L) + tmp7);
                                auto tmp14 = tmp13 >= tmp9;
                                auto tmp15 = tmp13 < tmp11;
                                auto tmp16 = tmp10 & tmp12;
                                auto tmp17 = tmp16 & tmp14;
                                auto tmp18 = tmp17 & tmp15;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = in_ptr2[static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*tmp7) + (5376L*tmp3) + (150528L*x0))];
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                                out_ptr3[static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0))] = tmp21;
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
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1382400L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = static_cast<float>(0.0);
                auto tmp1 = at::vec::Vectorized<float>(tmp0);
                tmp1.store(out_ptr1 + static_cast<long>(x0));
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
                                tmp1.store(out_ptr2 + static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x3) + (56448L*x0)));
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp0.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr2[static_cast<long>(x3 + (196L*x2) + (1764L*x1) + (1764L*x1_inner) + (56448L*x0))] = tmpbuf[x1_inner]; }
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                        tmp0.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (1764L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr2[static_cast<long>(x2 + (196L*x1) + (1764L*x0))];
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (1764L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    auto tmp0 = c10::convert<int>(1L + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(30);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = c10::convert<int>(1L + (static_cast<long>(x0) % static_cast<long>(28L)));
                    auto tmp6 = tmp5 >= tmp1;
                    auto tmp7 = tmp5 < tmp3;
                    auto tmp8 = tmp2 & tmp4;
                    auto tmp9 = tmp8 & tmp6;
                    auto tmp10 = tmp9 & tmp7;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(31L + (30L*(static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L))) + (900L*x1) + (900L*x1_inner) + (172800L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(28L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                        return tmp12;
                    }
                    ;
                    auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (5376L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_11 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (5376L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_permute_view_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (5376L*x3) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x3) + (5376L*x2) + (150528L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(192L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5376L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5568L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr2 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr2[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
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
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr1[static_cast<long>((14L*(c10::div_floor_integer(x3, 3L))) + (c10::div_floor_integer(x2, 14L)))];
                                auto tmp4 = in_ptr1[static_cast<long>((14L*(static_cast<long>(x3) % static_cast<long>(3L))) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = decltype(tmp0)(tmp0 + 30);
                                auto tmp2 = tmp0 < 0;
                                auto tmp3 = tmp2 ? tmp1 : tmp0;
                                TORCH_CHECK((0 <= tmp3) & (tmp3 < 30L), "index out of bounds: 0 <= tmp3 < 30L")
                                auto tmp5 = decltype(tmp4)(tmp4 + 30);
                                auto tmp6 = tmp4 < 0;
                                auto tmp7 = tmp6 ? tmp5 : tmp4;
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30L), "index out of bounds: 0 <= tmp7 < 30L")
                                auto tmp8 = c10::convert<long>((-1L) + tmp3);
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(28);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = c10::convert<long>((-1L) + tmp7);
                                auto tmp14 = tmp13 >= tmp9;
                                auto tmp15 = tmp13 < tmp11;
                                auto tmp16 = tmp10 & tmp12;
                                auto tmp17 = tmp16 & tmp14;
                                auto tmp18 = tmp17 & tmp15;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = in_ptr2[static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*tmp7) + (5376L*tmp3) + (150528L*x0))];
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                                out_ptr3[static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0))] = tmp21;
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
                       float* out_ptr0)
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
    }
}
''')


cpp_fused__unsafe_view_clone_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    auto tmp0 = c10::convert<int>(1L + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(30);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = c10::convert<int>(1L + (static_cast<long>(x0) % static_cast<long>(28L)));
                    auto tmp6 = tmp5 >= tmp1;
                    auto tmp7 = tmp5 < tmp3;
                    auto tmp8 = tmp2 & tmp4;
                    auto tmp9 = tmp8 & tmp6;
                    auto tmp10 = tmp9 & tmp7;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(31L + (30L*(static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L))) + (900L*x1) + (900L*x1_inner) + (172800L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(28L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                        return tmp12;
                    }
                    ;
                    auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_16 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (192L*x2) + (192L*x2_inner) + (5376L*x1) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (5376L*x2_inner) + (150528L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (150528L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_18 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (5376L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_permute_view_19 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (5376L*x3) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x3) + (5376L*x2) + (150528L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(192L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5376L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5568L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr2 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr2[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
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
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr1[static_cast<long>((14L*(c10::div_floor_integer(x3, 3L))) + (c10::div_floor_integer(x2, 14L)))];
                                auto tmp4 = in_ptr1[static_cast<long>((14L*(static_cast<long>(x3) % static_cast<long>(3L))) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = decltype(tmp0)(tmp0 + 30);
                                auto tmp2 = tmp0 < 0;
                                auto tmp3 = tmp2 ? tmp1 : tmp0;
                                TORCH_CHECK((0 <= tmp3) & (tmp3 < 30L), "index out of bounds: 0 <= tmp3 < 30L")
                                auto tmp5 = decltype(tmp4)(tmp4 + 30);
                                auto tmp6 = tmp4 < 0;
                                auto tmp7 = tmp6 ? tmp5 : tmp4;
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30L), "index out of bounds: 0 <= tmp7 < 30L")
                                auto tmp8 = c10::convert<long>((-1L) + tmp3);
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(28);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = c10::convert<long>((-1L) + tmp7);
                                auto tmp14 = tmp13 >= tmp9;
                                auto tmp15 = tmp13 < tmp11;
                                auto tmp16 = tmp10 & tmp12;
                                auto tmp17 = tmp16 & tmp14;
                                auto tmp18 = tmp17 & tmp15;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = in_ptr2[static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*tmp7) + (5376L*tmp3) + (150528L*x0))];
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                                out_ptr3[static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0))] = tmp21;
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
                       float* out_ptr0)
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
    }
}
''')


cpp_fused__unsafe_view_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    auto tmp0 = c10::convert<int>(1L + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(30);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = c10::convert<int>(1L + (static_cast<long>(x0) % static_cast<long>(28L)));
                    auto tmp6 = tmp5 >= tmp1;
                    auto tmp7 = tmp5 < tmp3;
                    auto tmp8 = tmp2 & tmp4;
                    auto tmp9 = tmp8 & tmp6;
                    auto tmp10 = tmp9 & tmp7;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(31L + (30L*(static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L))) + (900L*x1) + (900L*x1_inner) + (172800L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(28L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                        return tmp12;
                    }
                    ;
                    auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (5376L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_25 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (192L*x2) + (192L*x2_inner) + (5376L*x1) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (28L*x1) + (784L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (5376L*x2_inner) + (150528L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(192.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (28L*x3) + (5376L*x2) + (150528L*x0))] = tmp9;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_layer_norm_permute_view_26 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(28L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (5376L*x3) + (150528L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp2 = tmp0 * tmp1;
                            auto tmp4 = tmp2 + tmp3;
                            tmp4.store(out_ptr0 + static_cast<long>(x1 + (192L*x3) + (5376L*x2) + (150528L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(192L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5376L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(5568L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(14L))) + (10752L*(c10::div_floor_integer(x0, 14L)))));
                    auto tmp2 = tmp1 + tmp0;
                    auto tmp4 = tmp3 + tmp2;
                    auto tmp6 = tmp5 + tmp4;
                    auto tmp7 = static_cast<float>(0.25);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    tmp9.store(out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__softmax_clone_mul_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr2 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_out_ptr0[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = out_ptr1[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr2[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
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
                            #pragma GCC ivdep
                            for(long x4=static_cast<long>(0L); x4<static_cast<long>(32L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr1[static_cast<long>((14L*(c10::div_floor_integer(x3, 3L))) + (c10::div_floor_integer(x2, 14L)))];
                                auto tmp4 = in_ptr1[static_cast<long>((14L*(static_cast<long>(x3) % static_cast<long>(3L))) + (static_cast<long>(x2) % static_cast<long>(14L)))];
                                auto tmp1 = decltype(tmp0)(tmp0 + 30);
                                auto tmp2 = tmp0 < 0;
                                auto tmp3 = tmp2 ? tmp1 : tmp0;
                                TORCH_CHECK((0 <= tmp3) & (tmp3 < 30L), "index out of bounds: 0 <= tmp3 < 30L")
                                auto tmp5 = decltype(tmp4)(tmp4 + 30);
                                auto tmp6 = tmp4 < 0;
                                auto tmp7 = tmp6 ? tmp5 : tmp4;
                                TORCH_CHECK((0 <= tmp7) & (tmp7 < 30L), "index out of bounds: 0 <= tmp7 < 30L")
                                auto tmp8 = c10::convert<long>((-1L) + tmp3);
                                auto tmp9 = static_cast<long>(0);
                                auto tmp10 = tmp8 >= tmp9;
                                auto tmp11 = static_cast<long>(28);
                                auto tmp12 = tmp8 < tmp11;
                                auto tmp13 = c10::convert<long>((-1L) + tmp7);
                                auto tmp14 = tmp13 >= tmp9;
                                auto tmp15 = tmp13 < tmp11;
                                auto tmp16 = tmp10 & tmp12;
                                auto tmp17 = tmp16 & tmp14;
                                auto tmp18 = tmp17 & tmp15;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = in_ptr2[static_cast<long>((-5568L) + x4 + (32L*x1) + (192L*tmp7) + (5376L*tmp3) + (150528L*x0))];
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                                out_ptr3[static_cast<long>(x4 + (32L*x3) + (288L*x2) + (56448L*x1) + (338688L*x0))] = tmp21;
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
                       float* out_ptr0)
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
    }
}
''')


cpp_fused__unsafe_view_clone_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                    auto tmp0 = c10::convert<int>(1L + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)));
                    auto tmp1 = static_cast<int>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<int>(30);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = c10::convert<int>(1L + (static_cast<long>(x0) % static_cast<long>(28L)));
                    auto tmp6 = tmp5 >= tmp1;
                    auto tmp7 = tmp5 < tmp3;
                    auto tmp8 = tmp2 & tmp4;
                    auto tmp9 = tmp8 & tmp6;
                    auto tmp10 = tmp9 & tmp7;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(31L + (30L*(static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L))) + (900L*x1) + (900L*x1_inner) + (172800L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(x0) % static_cast<long>(28L)))]; return masked_load(tmpbuf, to_float_mask(tmp10)); })();
                        return tmp12;
                    }
                    ;
                    auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (192L*x2) + (5376L*x1) + (150528L*x0)));
                            auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (28L*x3) + (28L*x3_inner) + (5376L*x2) + (150528L*x0))] = tmpbuf[x3_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((28L*x1) + (28L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(28L))) + (150528L*(c10::div_floor_integer(x0, 784L))) + (static_cast<long>(c10::div_floor_integer(x0, 28L)) % static_cast<long>(28L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3612672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_permute_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
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
    }
}
''')


cpp_fused_add_native_layer_norm_view_33 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_36 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_54 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_55 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_60 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_66 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_67 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_72 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_75 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_78 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_81 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_84 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_85 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_90 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_91 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_93 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_96 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_97 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_98 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_99 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_102 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_103 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_105 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp6 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp13.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_108 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp8 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 * tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    tmp4.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_gelu_view_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_add_native_layer_norm_view_111 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0)));
                            auto tmp7 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp10 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (14L*x3_inner) + (5376L*x2) + (75264L*x0))] = tmpbuf[x3_inner]; }
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                    tmp3.store(out_ptr2 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    out_ptr2[static_cast<long>(x1 + (196L*x0))] = tmp2;
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (75264L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_view_114 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>((32L*(static_cast<long>(x0) % static_cast<long>(196L))) + (6272L*(c10::div_floor_integer((x1 + x1_inner), 32L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>((x1 + x1_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_115 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (384L*x2_inner) + (5376L*x1) + (75264L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (14L*x1) + (196L*x0)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp11.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (5376L*x2_inner) + (75264L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(384L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (384L*x2) + (5376L*x1) + (75264L*x0))];
                            auto tmp1 = out_ptr0[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp3 = out_ptr1[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp2 = decltype(tmp0)(tmp0 - tmp1);
                            auto tmp4 = static_cast<float>(384.0);
                            auto tmp5 = tmp3 / tmp4;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = 1 / std::sqrt(tmp7);
                            auto tmp9 = decltype(tmp2)(tmp2 * tmp8);
                            out_ptr2[static_cast<long>(x1 + (14L*x3) + (5376L*x2) + (75264L*x0))] = tmp9;
                        }
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
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr2[static_cast<long>((14L*x1) + (14L*x1_inner) + (5376L*(static_cast<long>(x0) % static_cast<long>(14L))) + (75264L*(c10::div_floor_integer(x0, 196L))) + (static_cast<long>(c10::div_floor_integer(x0, 14L)) % static_cast<long>(14L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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


cpp_fused_gelu_view_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1806336L); x0+=static_cast<long>(8L))
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


cpp_fused_cat_native_layer_norm_117 = async_compile.cpp('''
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
                        tmp17.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (75648L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6304L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_120 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75648L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75648L*x0)));
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
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_121 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_cat_native_layer_norm_122 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp11)::blendv(tmp17, tmp11, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
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
                       float* out_ptr2,
                       float* out_ptr3)
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
                            tmp0.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6304L*x1) + (75648L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_native_layer_norm_view_125 = async_compile.cpp('''
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75648L*x0)));
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
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (75648L*x0)));
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
                tmp13.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_gelu_view_126 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(9216L); x0+=static_cast<long>(8L))
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
''')


cpp_fused_cat_native_layer_norm_127 = async_compile.cpp('''
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(197L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = c10::convert<int>(x1);
                        auto tmp1 = static_cast<int>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<int>(1);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = masked_load(in_ptr0 + static_cast<long>(x2 + (75648L*x0)), to_float_mask(tmp4));
                            auto tmp7 = masked_load(in_ptr1 + static_cast<long>(x2 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp8 = tmp6 + tmp7;
                            auto tmp9 = masked_load(in_ptr2 + static_cast<long>(x2 + (384L*x0)), to_float_mask(tmp4));
                            auto tmp10 = tmp8 + tmp9;
                            return tmp10;
                        }
                        ;
                        auto tmp11 = decltype(tmp5())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp5(), to_float_mask(tmp4));
                        auto tmp12 = tmp0 >= tmp3;
                        auto tmp13 = static_cast<int>(197);
                        auto tmp14 = tmp0 < tmp13;
                        auto tmp15 = [&]
                        {
                            auto tmp16 = masked_load(in_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)), to_float_mask(tmp12));
                            return tmp16;
                        }
                        ;
                        auto tmp17 = decltype(tmp15())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp15(), to_float_mask(tmp12));
                        auto tmp18 = to_float_mask(tmp4);
                        auto tmp19 = decltype(tmp11)::blendv(tmp17, tmp11, tmp18);
                        tmp19.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (75648L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp10.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_128 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(384L + x1 + (384L*(static_cast<long>(x0) % static_cast<long>(196L))) + (75648L*(c10::div_floor_integer(x0, 196L)))));
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__softmax_add_detach_max_mul_native_layer_norm_native_layer_norm_backward_129 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       float* in_out_ptr3,
                       float* in_out_ptr4,
                       float* in_out_ptr5,
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
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const float* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const float* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const float* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const float* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const float* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const float* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const float* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const float* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const float* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const float* in_ptr61,
                       const float* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const float* in_ptr65,
                       const float* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const float* in_ptr69,
                       const float* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const float* in_ptr73,
                       const float* in_ptr74,
                       const float* in_ptr75,
                       const float* in_ptr76,
                       const float* in_ptr77,
                       const float* in_ptr78,
                       const float* in_ptr79,
                       const float* in_ptr80,
                       const long* in_ptr81,
                       const float* in_ptr82,
                       const float* in_ptr83,
                       const float* in_ptr84,
                       const long* in_ptr85,
                       const float* in_ptr86,
                       const float* in_ptr87,
                       const float* in_ptr88,
                       const long* in_ptr89,
                       float* out_ptr0,
                       long* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7,
                       float* out_ptr8,
                       float* out_ptr9,
                       float* out_ptr10,
                       float* out_ptr11,
                       float* out_ptr12,
                       float* out_ptr13,
                       float* out_ptr14,
                       float* out_ptr15,
                       float* out_ptr16,
                       float* out_ptr17,
                       float* out_ptr18,
                       float* out_ptr19,
                       float* out_ptr20,
                       float* out_ptr21,
                       float* out_ptr22,
                       float* out_ptr23,
                       float* out_ptr24,
                       float* out_ptr25,
                       float* out_ptr26,
                       float* out_ptr27,
                       float* out_ptr28,
                       float* out_ptr29,
                       float* out_ptr30,
                       float* out_ptr31,
                       float* out_ptr32,
                       float* out_ptr33,
                       float* out_ptr34,
                       float* out_ptr35,
                       float* out_ptr36,
                       float* out_ptr37,
                       float* out_ptr38,
                       float* out_ptr39,
                       float* out_ptr40,
                       float* out_ptr41,
                       float* out_ptr42,
                       float* out_ptr43,
                       float* out_ptr44,
                       float* out_ptr45,
                       float* out_ptr46,
                       float* out_ptr47,
                       float* out_ptr48,
                       float* out_ptr49,
                       float* out_ptr50,
                       float* out_ptr51,
                       float* out_ptr52,
                       float* out_ptr53,
                       float* out_ptr54,
                       float* out_ptr55,
                       float* out_ptr56,
                       float* out_ptr57,
                       float* out_ptr59,
                       float* out_ptr60,
                       long* out_ptr62,
                       float* out_ptr64,
                       float* out_ptr65,
                       long* out_ptr67,
                       float* out_ptr69,
                       float* out_ptr70,
                       long* out_ptr72)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1000L); x1+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        struct IndexValue_1 {size_t index; float value;};
                        IndexValue_1 tmp_acc1{0, -std::numeric_limits<float>::infinity()};
                        #if !defined(__clang_major__) || __clang_major__ > 9
                        #pragma omp declare reduction(argmax : IndexValue_1 :\
                            omp_out.value = omp_in.value < omp_out.value ? omp_out.value : omp_in.value,\
                            omp_out.index = omp_in.value < omp_out.value ? omp_out.index : omp_in.index)\
                        	initializer(omp_priv = {0, -std::numeric_limits<float>::infinity()})
                        #endif
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x1 + (1000L*x2) + (196000L*x0))];
                            auto tmp1 = in_ptr1[static_cast<long>(x1)];
                            auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp2);
                            if (tmp_acc1.value < tmp2) {
                                tmp_acc1.index = static_cast<long>(x2); tmp_acc1.value = tmp2;
                            }
                        }
                        out_ptr0[static_cast<long>(x1 + (1000L*x0))] = tmp_acc0;
                        out_ptr1[static_cast<long>(x1 + (1000L*x0))] = tmp_acc1.index;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = tmp0 + tmp4;
                    tmp5.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp8 = tmp7 / tmp2;
                    tmp8.store(in_out_ptr1 + static_cast<long>(x0));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (2364L*x0)));
                                auto tmp1 = in_ptr3[static_cast<long>(x1 + x1_inner + (12L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(tmp4 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp4, 8, out_ptr2 + static_cast<long>(x1 + (12L*x2) + (2364L*x0)), static_cast<long>(12L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (2364L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (12L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr2 + static_cast<long>(x1 + (12L*x2) + (2364L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp1 = in_ptr3[static_cast<long>(x1 + (12L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr2[static_cast<long>(x1 + (12L*x2) + (2364L*x0))] = tmp2;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(384.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
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
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            float tmp4[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (2364L*x0)));
                                auto tmp1 = in_ptr5[static_cast<long>(x1 + x1_inner + (12L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(tmp4 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp4, 8, out_ptr3 + static_cast<long>(x1 + (12L*x2) + (2364L*x0)), static_cast<long>(12L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (197L*x1) + (197L*x1_inner) + (2364L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (12L*x0)));
                            auto tmp2 = tmp0 / tmp1;
                            tmp2.store(out_ptr3 + static_cast<long>(x1 + (12L*x2) + (2364L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(197L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr4[static_cast<long>(x2 + (197L*x1) + (2364L*x0))];
                            auto tmp1 = in_ptr5[static_cast<long>(x1 + (12L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr3[static_cast<long>(x1 + (12L*x2) + (2364L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr4 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr4 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr6[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr4[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr8[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr5[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr7[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr8[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr5[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr6 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr6 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr9[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr6[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr7 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr7 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr10[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr7[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr12[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr8[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr11[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr12[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr8[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr9 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr13[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr9 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr13[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr9[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr14 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr10 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr14[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr10 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr14[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr10[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr16[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr11[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr15[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr16[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr11[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr17 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr12 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr17[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr12 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr17[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr12[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr18 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr13 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr18[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr13 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr18[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr13[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr20[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr14[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr19[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr20[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr14[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr21 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr15 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr21[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr15 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr21[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr15[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr22 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr16 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr22[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr16 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr22[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr16[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr24[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr17[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr23[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr24[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr17[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr25 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr18 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr25[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr18 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr25[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr18[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr26 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr19 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr26[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr19 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr26[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr19[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr28[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr20[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr27[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr28[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr20[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr29 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr21 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr29[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr21 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr29[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr21[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr30 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr22 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr30[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr22 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr30[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr22[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr32[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr23[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr31[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr32[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr23[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr33 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr24 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr33[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr24 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr33[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr24[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr34 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr25 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr34[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr25 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr34[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr25[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr36[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr26[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr35[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr36[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr26[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr37 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr27 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr37[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr27 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr37[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr27[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr38 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr28 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr38[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr28 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr38[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr28[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr40[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr29[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr39[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr40[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr29[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr41 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr30 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr41[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr30 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr41[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr30[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr42 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr31 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr42[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr31 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr42[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr31[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr44[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr32[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr43[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr44[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr32[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr45 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr33 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr45[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr33 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr45[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr33[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr46 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr34 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr46[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr34 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr46[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr34[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr48[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr35[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr47[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr48[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr35[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr49 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr36 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr49[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr36 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr49[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr36[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr50 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr37 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr50[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr37 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr50[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr37[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr52[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr38[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr51[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr52[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr38[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr53 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr39 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr53[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr39 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr53[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr39[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr54 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr40 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr54[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr40 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr54[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr40[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr56[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr41[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr55[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr56[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr41[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr57 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr42 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr57[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr42 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr57[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr42[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr58 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr43 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr58[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr43 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr58[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr43[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr59 + static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0)));
                            auto tmp1 = in_ptr60[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp3.store(tmpbuf); for (long x3_inner = 0; x3_inner < 8; x3_inner++) out_ptr44[static_cast<long>(x1 + (12L*x3) + (12L*x3_inner) + (2352L*x2) + (460992L*x0))] = tmpbuf[x3_inner]; }
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr59[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (460992L*x0))];
                            auto tmp1 = in_ptr60[static_cast<long>(x2 + (196L*x1) + (2352L*x0))];
                            auto tmp2 = tmp0 / tmp1;
                            out_ptr44[static_cast<long>(x1 + (12L*x3) + (2352L*x2) + (460992L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr61 + static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0)));
                                auto tmp1 = static_cast<float>(384.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr45 + static_cast<long>(x1 + (14L*x2) + (196L*x0)), static_cast<long>(14L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr61[static_cast<long>(x2 + (14L*x1) + (14L*x1_inner) + (196L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr45 + static_cast<long>(x1 + (14L*x2) + (196L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(8L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr61[static_cast<long>(x2 + (14L*x1) + (196L*x0))];
                            auto tmp1 = static_cast<float>(384.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr45[static_cast<long>(x1 + (14L*x2) + (196L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr62 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr46 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr62[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr46 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr62[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr46[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr63 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr64[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr47 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr63[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr64[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr47[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr65 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr48 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr65[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr48 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr65[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr48[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr66 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr49 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr66[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr49 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr66[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr49[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr67 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr68[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr50 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr67[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr68[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr50[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr69 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr51 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr69[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr51 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr69[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr51[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr70 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr52 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr70[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr52 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr70[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr52[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr71 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr72[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr53 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr71[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr72[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr53[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr73 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr54 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr73[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr54 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr73[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr54[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr74 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr55 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr74[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr55 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr74[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr55[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr75 + static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0)));
                                auto tmp1 = in_ptr76[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                tmp3.store(out_ptr56 + static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0)));
                            }
                            #pragma omp simd simdlen(4) 
                            for(long x4=static_cast<long>(8L); x4<static_cast<long>(9L); x4+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr75[static_cast<long>(x4 + (9L*x3) + (81L*x1) + (486L*x2) + (95256L*x0))];
                                auto tmp1 = in_ptr76[static_cast<long>(x3 + (9L*x2) + (1764L*x1) + (10584L*x0))];
                                auto tmp2 = tmp0 / tmp1;
                                out_ptr56[static_cast<long>(x4 + (9L*x3) + (81L*x2) + (15876L*x1) + (95256L*x0))] = tmp2;
                            }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp9[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr77 + static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0)));
                                auto tmp1 = static_cast<float>(192.0);
                                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                                auto tmp3 = tmp0 / tmp2;
                                auto tmp4 = static_cast<float>(1e-05);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 + tmp5;
                                auto tmp7 = tmp6.rsqrt();
                                auto tmp8 = tmp7 / tmp2;
                                tmp8.store(tmp9 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp9, 8, out_ptr57 + static_cast<long>(x1 + (28L*x2) + (784L*x0)), static_cast<long>(28L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr77[static_cast<long>(x2 + (28L*x1) + (28L*x1_inner) + (784L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp4 = static_cast<float>(1e-05);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 + tmp5;
                            auto tmp7 = tmp6.rsqrt();
                            auto tmp8 = tmp7 / tmp2;
                            tmp8.store(out_ptr57 + static_cast<long>(x1 + (28L*x2) + (784L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(24L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr77[static_cast<long>(x2 + (28L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(192.0);
                            auto tmp2 = tmp0 / tmp1;
                            auto tmp3 = static_cast<float>(1e-05);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            auto tmp5 = 1 / std::sqrt(tmp4);
                            auto tmp6 = tmp5 / tmp1;
                            out_ptr57[static_cast<long>(x1 + (28L*x2) + (784L*x0))] = tmp6;
                        }
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr78 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr59 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr59 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr81[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr62[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr82 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr64 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr64 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr85[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr67[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr86 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr69 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr69 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(100352.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.00000996502277);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr89[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr72[static_cast<long>(0L)] = tmp2;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261 = args
    args.clear()
    assert_size_stride(primals_1, (1, 14, 14, 384), (75264, 5376, 384, 1))
    assert_size_stride(primals_2, (1, 1, 384), (384, 384, 1))
    assert_size_stride(primals_3, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_7, (64, ), (1, ))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (192, 64, 4, 4), (1024, 16, 4, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, 192), (192, 1))
    assert_size_stride(primals_17, (486, 192), (192, 1))
    assert_size_stride(primals_18, (486, ), (1, ))
    assert_size_stride(primals_19, (192, 192), (192, 1))
    assert_size_stride(primals_20, (192, ), (1, ))
    assert_size_stride(primals_21, (192, ), (1, ))
    assert_size_stride(primals_22, (192, ), (1, ))
    assert_size_stride(primals_23, (576, 192), (192, 1))
    assert_size_stride(primals_24, (576, ), (1, ))
    assert_size_stride(primals_25, (192, 576), (576, 1))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, ), (1, ))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, 192), (192, 1))
    assert_size_stride(primals_30, (486, 192), (192, 1))
    assert_size_stride(primals_31, (486, ), (1, ))
    assert_size_stride(primals_32, (192, 192), (192, 1))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (576, 192), (192, 1))
    assert_size_stride(primals_37, (576, ), (1, ))
    assert_size_stride(primals_38, (192, 576), (576, 1))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (192, 192), (192, 1))
    assert_size_stride(primals_43, (486, 192), (192, 1))
    assert_size_stride(primals_44, (486, ), (1, ))
    assert_size_stride(primals_45, (192, 192), (192, 1))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (576, 192), (192, 1))
    assert_size_stride(primals_50, (576, ), (1, ))
    assert_size_stride(primals_51, (192, 576), (576, 1))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (192, ), (1, ))
    assert_size_stride(primals_54, (192, ), (1, ))
    assert_size_stride(primals_55, (192, 192), (192, 1))
    assert_size_stride(primals_56, (486, 192), (192, 1))
    assert_size_stride(primals_57, (486, ), (1, ))
    assert_size_stride(primals_58, (192, 192), (192, 1))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (576, 192), (192, 1))
    assert_size_stride(primals_63, (576, ), (1, ))
    assert_size_stride(primals_64, (192, 576), (576, 1))
    assert_size_stride(primals_65, (192, ), (1, ))
    assert_size_stride(primals_66, (384, 192, 2, 2), (768, 4, 2, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (1152, 384), (384, 1))
    assert_size_stride(primals_71, (384, 384), (384, 1))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (384, ), (1, ))
    assert_size_stride(primals_74, (384, ), (1, ))
    assert_size_stride(primals_75, (1152, 384), (384, 1))
    assert_size_stride(primals_76, (1152, ), (1, ))
    assert_size_stride(primals_77, (384, 1152), (1152, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (1152, 384), (384, 1))
    assert_size_stride(primals_82, (384, 384), (384, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (1152, 384), (384, 1))
    assert_size_stride(primals_87, (1152, ), (1, ))
    assert_size_stride(primals_88, (384, 1152), (1152, 1))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (1152, 384), (384, 1))
    assert_size_stride(primals_93, (384, 384), (384, 1))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (1152, 384), (384, 1))
    assert_size_stride(primals_98, (1152, ), (1, ))
    assert_size_stride(primals_99, (384, 1152), (1152, 1))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (1152, 384), (384, 1))
    assert_size_stride(primals_104, (384, 384), (384, 1))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (1152, 384), (384, 1))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_110, (384, 1152), (1152, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (1152, 384), (384, 1))
    assert_size_stride(primals_115, (384, 384), (384, 1))
    assert_size_stride(primals_116, (384, ), (1, ))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (1152, 384), (384, 1))
    assert_size_stride(primals_120, (1152, ), (1, ))
    assert_size_stride(primals_121, (384, 1152), (1152, 1))
    assert_size_stride(primals_122, (384, ), (1, ))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (1152, 384), (384, 1))
    assert_size_stride(primals_126, (384, 384), (384, 1))
    assert_size_stride(primals_127, (384, ), (1, ))
    assert_size_stride(primals_128, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (1152, 384), (384, 1))
    assert_size_stride(primals_131, (1152, ), (1, ))
    assert_size_stride(primals_132, (384, 1152), (1152, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (1152, 384), (384, 1))
    assert_size_stride(primals_137, (384, 384), (384, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (1152, 384), (384, 1))
    assert_size_stride(primals_142, (1152, ), (1, ))
    assert_size_stride(primals_143, (384, 1152), (1152, 1))
    assert_size_stride(primals_144, (384, ), (1, ))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (1152, 384), (384, 1))
    assert_size_stride(primals_148, (384, 384), (384, 1))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (1152, 384), (384, 1))
    assert_size_stride(primals_153, (1152, ), (1, ))
    assert_size_stride(primals_154, (384, 1152), (1152, 1))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (384, ), (1, ))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (1152, 384), (384, 1))
    assert_size_stride(primals_159, (384, 384), (384, 1))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (1152, 384), (384, 1))
    assert_size_stride(primals_164, (1152, ), (1, ))
    assert_size_stride(primals_165, (384, 1152), (1152, 1))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (1152, 384), (384, 1))
    assert_size_stride(primals_170, (384, 384), (384, 1))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (1152, 384), (384, 1))
    assert_size_stride(primals_175, (1152, ), (1, ))
    assert_size_stride(primals_176, (384, 1152), (1152, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (1152, 384), (384, 1))
    assert_size_stride(primals_181, (384, 384), (384, 1))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (1152, 384), (384, 1))
    assert_size_stride(primals_186, (1152, ), (1, ))
    assert_size_stride(primals_187, (384, 1152), (1152, 1))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (1152, 384), (384, 1))
    assert_size_stride(primals_192, (384, 384), (384, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (1152, 384), (384, 1))
    assert_size_stride(primals_197, (1152, ), (1, ))
    assert_size_stride(primals_198, (384, 1152), (1152, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (1152, 384), (384, 1))
    assert_size_stride(primals_203, (384, 384), (384, 1))
    assert_size_stride(primals_204, (384, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (1152, 384), (384, 1))
    assert_size_stride(primals_208, (1152, ), (1, ))
    assert_size_stride(primals_209, (384, 1152), (1152, 1))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (384, ), (1, ))
    assert_size_stride(primals_212, (384, ), (1, ))
    assert_size_stride(primals_213, (1152, 384), (384, 1))
    assert_size_stride(primals_214, (384, 384), (384, 1))
    assert_size_stride(primals_215, (384, ), (1, ))
    assert_size_stride(primals_216, (384, ), (1, ))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (1152, 384), (384, 1))
    assert_size_stride(primals_219, (1152, ), (1, ))
    assert_size_stride(primals_220, (384, 1152), (1152, 1))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (384, ), (1, ))
    assert_size_stride(primals_223, (384, ), (1, ))
    assert_size_stride(primals_224, (768, 384), (384, 1))
    assert_size_stride(primals_225, (384, 384), (384, 1))
    assert_size_stride(primals_226, (384, 384), (384, 1))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (384, ), (1, ))
    assert_size_stride(primals_229, (384, ), (1, ))
    assert_size_stride(primals_230, (1152, 384), (384, 1))
    assert_size_stride(primals_231, (1152, ), (1, ))
    assert_size_stride(primals_232, (384, 1152), (1152, 1))
    assert_size_stride(primals_233, (384, ), (1, ))
    assert_size_stride(primals_234, (384, ), (1, ))
    assert_size_stride(primals_235, (384, ), (1, ))
    assert_size_stride(primals_236, (768, 384), (384, 1))
    assert_size_stride(primals_237, (384, 384), (384, 1))
    assert_size_stride(primals_238, (384, 384), (384, 1))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (384, ), (1, ))
    assert_size_stride(primals_241, (384, ), (1, ))
    assert_size_stride(primals_242, (1152, 384), (384, 1))
    assert_size_stride(primals_243, (1152, ), (1, ))
    assert_size_stride(primals_244, (384, 1152), (1152, 1))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (384, ), (1, ))
    assert_size_stride(primals_247, (384, ), (1, ))
    assert_size_stride(primals_248, (1000, 384), (384, 1))
    assert_size_stride(primals_249, (1000, ), (1, ))
    assert_size_stride(primals_250, (1000, 384), (384, 1))
    assert_size_stride(primals_251, (1000, ), (1, ))
    assert_size_stride(primals_252, (64, ), (1, ))
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (64, ), (1, ))
    assert_size_stride(primals_256, (64, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((64, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((192, 64, 4, 4), (1024, 1, 256, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((384, 192, 2, 2), (768, 1, 384, 192), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_3.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()))
    del primals_12
    del primals_261
    del primals_3
    del primals_6
    del primals_66
    del primals_9
    # Source Nodes: [l__mod___patch_embed_conv_0], Original ATen: [aten.convolution]
    buf6 = extern_kernels.convolution(buf5, buf0, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 64, 112, 112), (802816, 1, 7168, 64))
    buf7 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf10 = empty((64, ), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_1(c_void_p(buf6.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_5
    # Source Nodes: [l__mod___patch_embed_conv_3], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (8, 64, 112, 112), (802816, 1, 7168, 64))
    buf13 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf16 = empty((64, ), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_2(c_void_p(buf12.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_8
    # Source Nodes: [l__mod___patch_embed_conv_6], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (8, 64, 112, 112), (802816, 1, 7168, 64))
    buf19 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf22 = empty((64, ), device='cpu', dtype=torch.float32)
    buf23 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_relu_3(c_void_p(buf18.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_11
    # Source Nodes: [x_1], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf23, buf3, primals_13, stride=(4, 4), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf24, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del primals_13
    buf25 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf29 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_view_4(c_void_p(buf24.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    buf30 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___0___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf29, reinterpret_tensor(primals_16, (192, 192), (1, 192), 0), out=buf30)
    buf31 = empty((3, 14), device='cpu', dtype=torch.int64)
    buf32 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf33 = empty((1568, 192), device='cpu', dtype=torch.float32)
    cpp_fused_im2col_native_layer_norm_permute_view_5(c_void_p(buf28.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_15
    buf34 = empty((1568, 486), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___0___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_18, buf33, reinterpret_tensor(primals_17, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf34)
    del primals_18
    buf35 = empty_strided((8, 6, 196, 9, 1), (10584, 9, 54, 1, 84672), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf34, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf34  # reuse
    buf37 = empty_strided((8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), device='cpu', dtype=torch.float32)
    buf38 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf39 = empty((8, 6, 196, 9, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_6(c_void_p(buf36.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = empty((9408, 9, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf39, (9408, 9, 32), (288, 32, 1), 0), out=buf40)
    buf41 = empty_strided((8, 192, 30, 30), (172800, 1, 5760, 192), device='cpu', dtype=torch.float32)
    buf42 = empty((8, 192, 30, 30), device='cpu', dtype=torch.float32)
    buf43 = empty((8, 6, 32, 9, 196), device='cpu', dtype=torch.float32)
    buf44 = reinterpret_tensor(buf43, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf43  # reuse
    cpp_fused_clone_col2im_7(c_void_p(buf44.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()))
    aten.index_put_(buf42, [None, None, reinterpret_tensor(buf31, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf31], buf44, True)
    buf47 = buf30; del buf30  # reuse
    cpp_fused__unsafe_view_clone_8(c_void_p(buf42.data_ptr()), c_void_p(buf47.data_ptr()))
    buf48 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf47, reinterpret_tensor(primals_19, (192, 192), (1, 192), 0), out=buf48)
    buf49 = buf25; del buf25  # reuse
    buf50 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf53 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_9(c_void_p(buf24.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_22
    buf54 = empty((6272, 576), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_24, buf53, reinterpret_tensor(primals_23, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf54)
    del primals_24
    buf55 = empty((6272, 576), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_10(c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()))
    buf56 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_26, buf55, reinterpret_tensor(primals_25, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf56)
    del primals_26
    buf57 = buf49; del buf49  # reuse
    buf58 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf61 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_11(c_void_p(buf24.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___1___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf61, reinterpret_tensor(primals_29, (192, 192), (1, 192), 0), out=buf62)
    buf63 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf64 = empty((1568, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_view_12(c_void_p(buf60.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()))
    del primals_28
    buf65 = empty((1568, 486), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___1___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_31, buf64, reinterpret_tensor(primals_30, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf65)
    del primals_31
    buf66 = buf35; del buf35  # reuse
    buf67 = reinterpret_tensor(buf65, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf65  # reuse
    buf68 = empty_strided((8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), device='cpu', dtype=torch.float32)
    buf69 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf70 = reinterpret_tensor(buf44, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf44  # reuse
    cpp_fused__softmax_clone_mul_13(c_void_p(buf67.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = buf40; del buf40  # reuse
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf69, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf70, (9408, 9, 32), (288, 32, 1), 0), out=buf71)
    buf72 = buf42; del buf42  # reuse
    buf73 = empty((8, 6, 32, 9, 196), device='cpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf73, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf73  # reuse
    cpp_fused_clone_col2im_14(c_void_p(buf74.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    aten.index_put_(buf72, [None, None, reinterpret_tensor(buf31, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf31], buf74, True)
    buf77 = buf62; del buf62  # reuse
    cpp_fused__unsafe_view_clone_15(c_void_p(buf72.data_ptr()), c_void_p(buf77.data_ptr()))
    buf78 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf77, reinterpret_tensor(primals_32, (192, 192), (1, 192), 0), out=buf78)
    buf79 = reinterpret_tensor(buf78, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf78  # reuse
    buf80 = buf57; del buf57  # reuse
    buf81 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf83 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf84 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_16(c_void_p(buf79.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del primals_20
    del primals_33
    del primals_35
    buf85 = empty((6272, 576), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_37, buf84, reinterpret_tensor(primals_36, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf85)
    del primals_37
    buf86 = empty((6272, 576), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_17(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()))
    buf87 = buf56; del buf56  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_39, buf86, reinterpret_tensor(primals_38, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf87)
    del primals_39
    buf88 = buf80; del buf80  # reuse
    buf89 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf91 = reinterpret_tensor(buf48, (8, 28, 28, 192), (150528, 1, 5376, 28), 0); del buf48  # reuse
    buf92 = reinterpret_tensor(buf24, (6272, 192), (192, 1), 0); del buf24  # reuse
    cpp_fused_add_native_layer_norm_view_18(c_void_p(buf79.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    buf93 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___2___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf92, reinterpret_tensor(primals_42, (192, 192), (1, 192), 0), out=buf93)
    buf94 = empty_strided((8, 192, 28, 28), (150528, 1, 5376, 192), device='cpu', dtype=torch.float32)
    buf95 = empty((1568, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_view_19(c_void_p(buf91.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_41
    buf96 = empty((1568, 486), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___2___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_44, buf95, reinterpret_tensor(primals_43, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf96)
    del primals_44
    buf97 = buf66; del buf66  # reuse
    buf98 = reinterpret_tensor(buf96, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf96  # reuse
    buf99 = empty_strided((8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), device='cpu', dtype=torch.float32)
    buf100 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf101 = reinterpret_tensor(buf74, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf74  # reuse
    cpp_fused__softmax_clone_mul_20(c_void_p(buf98.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    buf102 = buf71; del buf71  # reuse
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf100, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf101, (9408, 9, 32), (288, 32, 1), 0), out=buf102)
    buf103 = buf72; del buf72  # reuse
    buf104 = empty((8, 6, 32, 9, 196), device='cpu', dtype=torch.float32)
    buf105 = reinterpret_tensor(buf104, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf104  # reuse
    cpp_fused_clone_col2im_21(c_void_p(buf105.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    aten.index_put_(buf103, [None, None, reinterpret_tensor(buf31, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf31], buf105, True)
    buf108 = buf93; del buf93  # reuse
    cpp_fused__unsafe_view_clone_22(c_void_p(buf103.data_ptr()), c_void_p(buf108.data_ptr()))
    buf109 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf108, reinterpret_tensor(primals_45, (192, 192), (1, 192), 0), out=buf109)
    buf110 = buf88; del buf88  # reuse
    buf111 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf113 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf114 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_23(c_void_p(buf79.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del primals_48
    buf115 = empty((6272, 576), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_50, buf114, reinterpret_tensor(primals_49, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf115)
    del primals_50
    buf116 = empty((6272, 576), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_24(c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()))
    buf117 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_36], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_52, buf116, reinterpret_tensor(primals_51, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf117)
    del primals_52
    buf118 = reinterpret_tensor(buf117, (8, 28, 28, 192), (150528, 5376, 192, 1), 0); del buf117  # reuse
    buf119 = buf110; del buf110  # reuse
    buf120 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf122 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf123 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_25(c_void_p(buf118.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf119.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(buf123.data_ptr()))
    del primals_46
    buf124 = buf87; del buf87  # reuse
    # Source Nodes: [getattr_l__mod___network_0___3___attn_v], Original ATen: [aten.mm]
    extern_kernels.mm(buf123, reinterpret_tensor(primals_55, (192, 192), (1, 192), 0), out=buf124)
    buf125 = reinterpret_tensor(buf79, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf79  # reuse
    buf126 = empty((1568, 192), device='cpu', dtype=torch.float32)
    cpp_fused_native_layer_norm_permute_view_26(c_void_p(buf122.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del primals_54
    buf127 = empty((1568, 486), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_0___3___attn_attn], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_57, buf126, reinterpret_tensor(primals_56, (192, 486), (1, 192), 0), alpha=1, beta=1, out=buf127)
    del primals_57
    buf128 = buf97; del buf97  # reuse
    buf129 = reinterpret_tensor(buf127, (8, 6, 196, 9, 9), (95256, 81, 486, 9, 1), 0); del buf127  # reuse
    buf130 = empty_strided((8, 6, 196, 9, 1), (10584, 1764, 9, 1, 84672), device='cpu', dtype=torch.float32)
    buf131 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf132 = reinterpret_tensor(buf105, (8, 6, 196, 9, 32), (338688, 56448, 288, 32, 1), 0); del buf105  # reuse
    cpp_fused__softmax_clone_mul_27(c_void_p(buf129.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()))
    del buf128
    buf133 = buf102; del buf102  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf131, (9408, 9, 9), (81, 9, 1), 0), reinterpret_tensor(buf132, (9408, 9, 32), (288, 32, 1), 0), out=buf133)
    buf134 = buf103; del buf103  # reuse
    buf135 = empty((8, 6, 32, 9, 196), device='cpu', dtype=torch.float32)
    buf136 = reinterpret_tensor(buf135, (8, 192, 3, 14, 3, 14), (338688, 1764, 588, 14, 196, 1), 0); del buf135  # reuse
    cpp_fused_clone_col2im_28(c_void_p(buf136.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    del buf133
    aten.index_put_(buf134, [None, None, reinterpret_tensor(buf31, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf31], buf136, True)
    del buf136
    buf139 = buf124; del buf124  # reuse
    cpp_fused__unsafe_view_clone_29(c_void_p(buf134.data_ptr()), c_void_p(buf139.data_ptr()))
    del buf134
    buf140 = buf109; del buf109  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.mm]
    extern_kernels.mm(buf139, reinterpret_tensor(primals_58, (192, 192), (1, 192), 0), out=buf140)
    buf141 = buf119; del buf119  # reuse
    buf142 = empty_strided((8, 28, 28, 1), (784, 28, 1, 6272), device='cpu', dtype=torch.float32)
    buf144 = empty_strided((8, 28, 28, 192), (150528, 1, 5376, 28), device='cpu', dtype=torch.float32)
    buf145 = empty((6272, 192), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_30(c_void_p(buf118.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    del primals_61
    buf146 = empty((6272, 576), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_63, buf145, reinterpret_tensor(primals_62, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf146)
    del primals_63
    buf147 = empty((6272, 576), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_31(c_void_p(buf146.data_ptr()), c_void_p(buf147.data_ptr()))
    buf148 = empty((6272, 192), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_48], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_65, buf147, reinterpret_tensor(primals_64, (576, 192), (1, 576), 0), alpha=1, beta=1, out=buf148)
    del primals_65
    buf149 = reinterpret_tensor(buf148, (8, 192, 28, 28), (150528, 1, 5376, 192), 0); del buf148  # reuse
    cpp_fused_add_permute_32(c_void_p(buf149.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_59.data_ptr()))
    del buf118
    del buf140
    del primals_59
    # Source Nodes: [x_53], Original ATen: [aten.convolution]
    buf150 = extern_kernels.convolution(buf149, buf4, primals_67, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf150, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del primals_67
    buf151 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf152 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf154 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf155 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_33(c_void_p(buf150.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del primals_69
    buf156 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_2___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf155, reinterpret_tensor(primals_70, (384, 1152), (1, 384), 0), out=buf156)
    buf157 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf158 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_34(c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    buf159 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf157, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf158, (96, 32, 196), (6272, 196, 1), 0), out=buf159)
    buf160 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf161 = reinterpret_tensor(buf159, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf159  # reuse
    buf162 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf163 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf164 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_35(c_void_p(buf161.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()))
    buf165 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf163, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf164, (96, 196, 32), (6272, 32, 1), 0), out=buf165)
    buf166 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_36(c_void_p(buf165.data_ptr()), c_void_p(buf166.data_ptr()))
    buf167 = reinterpret_tensor(buf165, (1568, 384), (384, 1), 0); del buf165  # reuse
    # Source Nodes: [x_59], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_72, buf166, reinterpret_tensor(primals_71, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf167)
    del primals_72
    buf168 = buf151; del buf151  # reuse
    buf169 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf171 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf172 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_37(c_void_p(buf150.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del primals_74
    buf173 = buf156; del buf156  # reuse
    # Source Nodes: [x_62], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_76, buf172, reinterpret_tensor(primals_75, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf173)
    del primals_76
    buf174 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_38(c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    buf175 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_66], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_78, buf174, reinterpret_tensor(primals_77, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf175)
    del primals_78
    buf176 = buf168; del buf168  # reuse
    buf177 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf179 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf180 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_39(c_void_p(buf150.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()))
    del primals_80
    buf181 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_2___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf180, reinterpret_tensor(primals_81, (384, 1152), (1, 384), 0), out=buf181)
    buf182 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf183 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_40(c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()))
    buf184 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf182, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf183, (96, 32, 196), (6272, 196, 1), 0), out=buf184)
    buf185 = buf160; del buf160  # reuse
    buf186 = reinterpret_tensor(buf184, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf184  # reuse
    buf187 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf188 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf189 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_41(c_void_p(buf186.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()))
    buf190 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf188, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf189, (96, 196, 32), (6272, 32, 1), 0), out=buf190)
    buf191 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_42(c_void_p(buf190.data_ptr()), c_void_p(buf191.data_ptr()))
    buf192 = reinterpret_tensor(buf190, (1568, 384), (384, 1), 0); del buf190  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_83, buf191, reinterpret_tensor(primals_82, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf192)
    del primals_83
    buf193 = reinterpret_tensor(buf192, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf192  # reuse
    buf194 = buf176; del buf176  # reuse
    buf195 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf197 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf198 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_43(c_void_p(buf193.data_ptr()), c_void_p(buf150.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del primals_1
    del primals_85
    buf199 = buf181; del buf181  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_87, buf198, reinterpret_tensor(primals_86, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf199)
    del primals_87
    buf200 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_44(c_void_p(buf199.data_ptr()), c_void_p(buf200.data_ptr()))
    buf201 = buf175; del buf175  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_89, buf200, reinterpret_tensor(primals_88, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf201)
    del primals_89
    buf202 = buf194; del buf194  # reuse
    buf203 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf205 = reinterpret_tensor(buf167, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf167  # reuse
    buf206 = reinterpret_tensor(buf150, (1568, 384), (384, 1), 0); del buf150  # reuse
    cpp_fused_add_native_layer_norm_view_45(c_void_p(buf193.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf202.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del primals_91
    buf207 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_2___2___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf206, reinterpret_tensor(primals_92, (384, 1152), (1, 384), 0), out=buf207)
    buf208 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf209 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_46(c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf209.data_ptr()))
    buf210 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf209, (96, 32, 196), (6272, 196, 1), 0), out=buf210)
    buf211 = buf185; del buf185  # reuse
    buf212 = reinterpret_tensor(buf210, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf210  # reuse
    buf213 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf214 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf215 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_47(c_void_p(buf212.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    buf216 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf214, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf215, (96, 196, 32), (6272, 32, 1), 0), out=buf216)
    buf217 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_48(c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()))
    buf218 = reinterpret_tensor(buf216, (1568, 384), (384, 1), 0); del buf216  # reuse
    # Source Nodes: [x_81], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_94, buf217, reinterpret_tensor(primals_93, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf218)
    del primals_94
    buf219 = buf202; del buf202  # reuse
    buf220 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf222 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf223 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_49(c_void_p(buf193.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()))
    del primals_96
    buf224 = buf207; del buf207  # reuse
    # Source Nodes: [x_84], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_98, buf223, reinterpret_tensor(primals_97, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf224)
    del primals_98
    buf225 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_50(c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()))
    buf226 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_88], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_100, buf225, reinterpret_tensor(primals_99, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf226)
    del primals_100
    buf227 = buf219; del buf219  # reuse
    buf228 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf230 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf231 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_51(c_void_p(buf193.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    del primals_102
    buf232 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_2___3___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf231, reinterpret_tensor(primals_103, (384, 1152), (1, 384), 0), out=buf232)
    buf233 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf234 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_52(c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()))
    buf235 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf234, (96, 32, 196), (6272, 196, 1), 0), out=buf235)
    buf236 = buf211; del buf211  # reuse
    buf237 = reinterpret_tensor(buf235, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf235  # reuse
    buf238 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf239 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf240 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_53(c_void_p(buf237.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf239, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf240, (96, 196, 32), (6272, 32, 1), 0), out=buf241)
    buf242 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_54(c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    buf243 = reinterpret_tensor(buf241, (1568, 384), (384, 1), 0); del buf241  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_105, buf242, reinterpret_tensor(primals_104, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf243)
    del primals_105
    buf244 = reinterpret_tensor(buf243, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf243  # reuse
    buf245 = buf227; del buf227  # reuse
    buf246 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf248 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf249 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_55(c_void_p(buf244.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    del primals_107
    buf250 = buf232; del buf232  # reuse
    # Source Nodes: [x_95], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_109, buf249, reinterpret_tensor(primals_108, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf250)
    del primals_109
    buf251 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_56(c_void_p(buf250.data_ptr()), c_void_p(buf251.data_ptr()))
    buf252 = buf226; del buf226  # reuse
    # Source Nodes: [x_99], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_111, buf251, reinterpret_tensor(primals_110, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf252)
    del primals_111
    buf253 = buf245; del buf245  # reuse
    buf254 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf256 = reinterpret_tensor(buf218, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf218  # reuse
    buf257 = buf201; del buf201  # reuse
    cpp_fused_add_native_layer_norm_view_57(c_void_p(buf244.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()))
    del primals_113
    buf258 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf257, reinterpret_tensor(primals_114, (384, 1152), (1, 384), 0), out=buf258)
    buf259 = reinterpret_tensor(buf193, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf193  # reuse
    buf260 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_58(c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()))
    buf261 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf259, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf260, (96, 32, 196), (6272, 196, 1), 0), out=buf261)
    buf262 = buf236; del buf236  # reuse
    buf263 = reinterpret_tensor(buf261, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf261  # reuse
    buf264 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf265 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf266 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_59(c_void_p(buf263.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf266.data_ptr()))
    buf267 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf266, (96, 196, 32), (6272, 32, 1), 0), out=buf267)
    buf268 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_60(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (1568, 384), (384, 1), 0); del buf267  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_116, buf268, reinterpret_tensor(primals_115, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf269)
    del primals_116
    buf270 = buf253; del buf253  # reuse
    buf271 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf273 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf274 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_61(c_void_p(buf244.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(buf274.data_ptr()))
    del primals_118
    buf275 = buf258; del buf258  # reuse
    # Source Nodes: [x_107], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_120, buf274, reinterpret_tensor(primals_119, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf275)
    del primals_120
    buf276 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_62(c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()))
    buf277 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_111], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_122, buf276, reinterpret_tensor(primals_121, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf277)
    del primals_122
    buf278 = buf270; del buf270  # reuse
    buf279 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf281 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf282 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_63(c_void_p(buf244.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf282.data_ptr()))
    del primals_124
    buf283 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf282, reinterpret_tensor(primals_125, (384, 1152), (1, 384), 0), out=buf283)
    buf284 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf285 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_64(c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    buf286 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf285, (96, 32, 196), (6272, 196, 1), 0), out=buf286)
    buf287 = buf262; del buf262  # reuse
    buf288 = reinterpret_tensor(buf286, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf286  # reuse
    buf289 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf290 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf291 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_65(c_void_p(buf288.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    buf292 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf290, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf291, (96, 196, 32), (6272, 32, 1), 0), out=buf292)
    buf293 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_66(c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    buf294 = reinterpret_tensor(buf292, (1568, 384), (384, 1), 0); del buf292  # reuse
    # Source Nodes: [x_115], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_127, buf293, reinterpret_tensor(primals_126, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf294)
    del primals_127
    buf295 = reinterpret_tensor(buf294, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf294  # reuse
    buf296 = buf278; del buf278  # reuse
    buf297 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf299 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf300 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_67(c_void_p(buf295.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()))
    del primals_129
    buf301 = buf283; del buf283  # reuse
    # Source Nodes: [x_118], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_131, buf300, reinterpret_tensor(primals_130, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf301)
    del primals_131
    buf302 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_68(c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    buf303 = buf277; del buf277  # reuse
    # Source Nodes: [x_122], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_133, buf302, reinterpret_tensor(primals_132, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf303)
    del primals_133
    buf304 = buf296; del buf296  # reuse
    buf305 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf307 = reinterpret_tensor(buf269, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf269  # reuse
    buf308 = buf252; del buf252  # reuse
    cpp_fused_add_native_layer_norm_view_69(c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()))
    del primals_135
    buf309 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___2___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf308, reinterpret_tensor(primals_136, (384, 1152), (1, 384), 0), out=buf309)
    buf310 = reinterpret_tensor(buf244, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf244  # reuse
    buf311 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_70(c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()))
    buf312 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf310, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf311, (96, 32, 196), (6272, 196, 1), 0), out=buf312)
    buf313 = buf287; del buf287  # reuse
    buf314 = reinterpret_tensor(buf312, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf312  # reuse
    buf315 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf316 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf317 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_71(c_void_p(buf314.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()))
    buf318 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf316, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf317, (96, 196, 32), (6272, 32, 1), 0), out=buf318)
    buf319 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_72(c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()))
    buf320 = reinterpret_tensor(buf318, (1568, 384), (384, 1), 0); del buf318  # reuse
    # Source Nodes: [x_126], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_138, buf319, reinterpret_tensor(primals_137, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf320)
    del primals_138
    buf321 = buf304; del buf304  # reuse
    buf322 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf324 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf325 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_73(c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del primals_140
    buf326 = buf309; del buf309  # reuse
    # Source Nodes: [x_129], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_142, buf325, reinterpret_tensor(primals_141, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf326)
    del primals_142
    buf327 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_74(c_void_p(buf326.data_ptr()), c_void_p(buf327.data_ptr()))
    buf328 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_133], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_144, buf327, reinterpret_tensor(primals_143, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf328)
    del primals_144
    buf329 = buf321; del buf321  # reuse
    buf330 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf332 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf333 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_75(c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del primals_146
    buf334 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___3___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf333, reinterpret_tensor(primals_147, (384, 1152), (1, 384), 0), out=buf334)
    buf335 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf336 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_76(c_void_p(buf334.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf336.data_ptr()))
    buf337 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf335, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf336, (96, 32, 196), (6272, 196, 1), 0), out=buf337)
    buf338 = buf313; del buf313  # reuse
    buf339 = reinterpret_tensor(buf337, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf337  # reuse
    buf340 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf341 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf342 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_77(c_void_p(buf339.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf341, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf342, (96, 196, 32), (6272, 32, 1), 0), out=buf343)
    buf344 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_78(c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    buf345 = reinterpret_tensor(buf343, (1568, 384), (384, 1), 0); del buf343  # reuse
    # Source Nodes: [x_137], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_149, buf344, reinterpret_tensor(primals_148, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf345)
    del primals_149
    buf346 = reinterpret_tensor(buf345, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf345  # reuse
    buf347 = buf329; del buf329  # reuse
    buf348 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf350 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf351 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_79(c_void_p(buf346.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()))
    del primals_151
    buf352 = buf334; del buf334  # reuse
    # Source Nodes: [x_140], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_153, buf351, reinterpret_tensor(primals_152, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf352)
    del primals_153
    buf353 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_80(c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()))
    buf354 = buf328; del buf328  # reuse
    # Source Nodes: [x_144], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_155, buf353, reinterpret_tensor(primals_154, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf354)
    del primals_155
    buf355 = buf347; del buf347  # reuse
    buf356 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf358 = reinterpret_tensor(buf320, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf320  # reuse
    buf359 = buf303; del buf303  # reuse
    cpp_fused_add_native_layer_norm_view_81(c_void_p(buf346.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf359.data_ptr()))
    del primals_157
    buf360 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___4___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf359, reinterpret_tensor(primals_158, (384, 1152), (1, 384), 0), out=buf360)
    buf361 = reinterpret_tensor(buf295, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf295  # reuse
    buf362 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_82(c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf362.data_ptr()))
    buf363 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf361, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf362, (96, 32, 196), (6272, 196, 1), 0), out=buf363)
    buf364 = buf338; del buf338  # reuse
    buf365 = reinterpret_tensor(buf363, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf363  # reuse
    buf366 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf367 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf368 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_83(c_void_p(buf365.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf368.data_ptr()))
    buf369 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf367, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf368, (96, 196, 32), (6272, 32, 1), 0), out=buf369)
    buf370 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_84(c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()))
    buf371 = reinterpret_tensor(buf369, (1568, 384), (384, 1), 0); del buf369  # reuse
    # Source Nodes: [x_148], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_160, buf370, reinterpret_tensor(primals_159, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf371)
    del primals_160
    buf372 = buf355; del buf355  # reuse
    buf373 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf375 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf376 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_85(c_void_p(buf346.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(primals_162.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()))
    del primals_162
    buf377 = buf360; del buf360  # reuse
    # Source Nodes: [x_151], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_164, buf376, reinterpret_tensor(primals_163, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf377)
    del primals_164
    buf378 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_86(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()))
    buf379 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_155], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_166, buf378, reinterpret_tensor(primals_165, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf379)
    del primals_166
    buf380 = buf372; del buf372  # reuse
    buf381 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf383 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf384 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_87(c_void_p(buf346.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(primals_168.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf383.data_ptr()), c_void_p(buf384.data_ptr()))
    del primals_168
    buf385 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___5___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf384, reinterpret_tensor(primals_169, (384, 1152), (1, 384), 0), out=buf385)
    buf386 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf387 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_88(c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf387.data_ptr()))
    buf388 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf386, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf387, (96, 32, 196), (6272, 196, 1), 0), out=buf388)
    buf389 = buf364; del buf364  # reuse
    buf390 = reinterpret_tensor(buf388, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf388  # reuse
    buf391 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf392 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf393 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_89(c_void_p(buf390.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()))
    buf394 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf392, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf393, (96, 196, 32), (6272, 32, 1), 0), out=buf394)
    buf395 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_90(c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    buf396 = reinterpret_tensor(buf394, (1568, 384), (384, 1), 0); del buf394  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_171, buf395, reinterpret_tensor(primals_170, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf396)
    del primals_171
    buf397 = reinterpret_tensor(buf396, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf396  # reuse
    buf398 = buf380; del buf380  # reuse
    buf399 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf401 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf402 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_91(c_void_p(buf397.data_ptr()), c_void_p(buf346.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del primals_173
    buf403 = buf385; del buf385  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_175, buf402, reinterpret_tensor(primals_174, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf403)
    del primals_175
    buf404 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_92(c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()))
    buf405 = buf379; del buf379  # reuse
    # Source Nodes: [x_166], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_177, buf404, reinterpret_tensor(primals_176, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf405)
    del primals_177
    buf406 = buf398; del buf398  # reuse
    buf407 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf409 = reinterpret_tensor(buf371, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf371  # reuse
    buf410 = buf354; del buf354  # reuse
    cpp_fused_add_native_layer_norm_view_93(c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf406.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(buf410.data_ptr()))
    del primals_179
    buf411 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___6___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf410, reinterpret_tensor(primals_180, (384, 1152), (1, 384), 0), out=buf411)
    buf412 = reinterpret_tensor(buf346, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf346  # reuse
    buf413 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_94(c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()))
    buf414 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf412, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf413, (96, 32, 196), (6272, 196, 1), 0), out=buf414)
    buf415 = buf389; del buf389  # reuse
    buf416 = reinterpret_tensor(buf414, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf414  # reuse
    buf417 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf418 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf419 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_95(c_void_p(buf416.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf415.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()), c_void_p(buf419.data_ptr()))
    buf420 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf418, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf419, (96, 196, 32), (6272, 32, 1), 0), out=buf420)
    buf421 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_96(c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    buf422 = reinterpret_tensor(buf420, (1568, 384), (384, 1), 0); del buf420  # reuse
    # Source Nodes: [x_170], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_182, buf421, reinterpret_tensor(primals_181, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf422)
    del primals_182
    buf423 = buf406; del buf406  # reuse
    buf424 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf426 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf427 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_97(c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()))
    del primals_184
    buf428 = buf411; del buf411  # reuse
    # Source Nodes: [x_173], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_186, buf427, reinterpret_tensor(primals_185, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf428)
    del primals_186
    buf429 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_98(c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()))
    buf430 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_177], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_188, buf429, reinterpret_tensor(primals_187, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf430)
    del primals_188
    buf431 = buf423; del buf423  # reuse
    buf432 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf434 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf435 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_99(c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_189.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf435.data_ptr()))
    del primals_190
    buf436 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_3___7___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf435, reinterpret_tensor(primals_191, (384, 1152), (1, 384), 0), out=buf436)
    buf437 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf438 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_100(c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf438.data_ptr()))
    buf439 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf437, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf438, (96, 32, 196), (6272, 196, 1), 0), out=buf439)
    buf440 = buf415; del buf415  # reuse
    buf441 = reinterpret_tensor(buf439, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf439  # reuse
    buf442 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf443 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf444 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_101(c_void_p(buf441.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf443, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf444, (96, 196, 32), (6272, 32, 1), 0), out=buf445)
    buf446 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_102(c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()))
    buf447 = reinterpret_tensor(buf445, (1568, 384), (384, 1), 0); del buf445  # reuse
    # Source Nodes: [x_181], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_193, buf446, reinterpret_tensor(primals_192, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf447)
    del primals_193
    buf448 = reinterpret_tensor(buf447, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf447  # reuse
    buf449 = buf431; del buf431  # reuse
    buf450 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf452 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf453 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_103(c_void_p(buf448.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf422.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del primals_195
    buf454 = buf436; del buf436  # reuse
    # Source Nodes: [x_184], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_197, buf453, reinterpret_tensor(primals_196, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf454)
    del primals_197
    buf455 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_104(c_void_p(buf454.data_ptr()), c_void_p(buf455.data_ptr()))
    buf456 = buf430; del buf430  # reuse
    # Source Nodes: [x_188], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_199, buf455, reinterpret_tensor(primals_198, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf456)
    del primals_199
    buf457 = buf449; del buf449  # reuse
    buf458 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf460 = reinterpret_tensor(buf422, (8, 14, 14, 384), (75264, 1, 5376, 14), 0); del buf422  # reuse
    buf461 = buf405; del buf405  # reuse
    cpp_fused_add_native_layer_norm_view_105(c_void_p(buf448.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(buf457.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()))
    del primals_201
    buf462 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_4___0___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf461, reinterpret_tensor(primals_202, (384, 1152), (1, 384), 0), out=buf462)
    buf463 = reinterpret_tensor(buf397, (8, 12, 196, 32), (75264, 6272, 32, 1), 0); del buf397  # reuse
    buf464 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_106(c_void_p(buf462.data_ptr()), c_void_p(buf463.data_ptr()), c_void_p(buf464.data_ptr()))
    buf465 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf463, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf464, (96, 32, 196), (6272, 196, 1), 0), out=buf465)
    buf466 = buf440; del buf440  # reuse
    buf467 = reinterpret_tensor(buf465, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf465  # reuse
    buf468 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf469 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf470 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_107(c_void_p(buf467.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf466.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    buf471 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf469, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf470, (96, 196, 32), (6272, 32, 1), 0), out=buf471)
    buf472 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_108(c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    buf473 = reinterpret_tensor(buf471, (1568, 384), (384, 1), 0); del buf471  # reuse
    # Source Nodes: [x_193], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf472, reinterpret_tensor(primals_203, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf473)
    del primals_204
    buf474 = buf457; del buf457  # reuse
    buf475 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf477 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf478 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_109(c_void_p(buf448.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    del primals_206
    buf479 = buf462; del buf462  # reuse
    # Source Nodes: [x_196], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf478, reinterpret_tensor(primals_207, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf479)
    del primals_208
    buf480 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_110(c_void_p(buf479.data_ptr()), c_void_p(buf480.data_ptr()))
    buf481 = empty((1568, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_200], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_210, buf480, reinterpret_tensor(primals_209, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf481)
    del primals_210
    buf482 = buf474; del buf474  # reuse
    buf483 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf485 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf486 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_111(c_void_p(buf448.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf482.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()), c_void_p(buf486.data_ptr()))
    del primals_212
    buf487 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [getattr_l__mod___network_4___1___attn_qkv], Original ATen: [aten.mm]
    extern_kernels.mm(buf486, reinterpret_tensor(primals_213, (384, 1152), (1, 384), 0), out=buf487)
    buf488 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    buf489 = empty((8, 12, 32, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_112(c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()), c_void_p(buf489.data_ptr()))
    buf490 = empty((96, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf488, (96, 196, 32), (6272, 32, 1), 0), reinterpret_tensor(buf489, (96, 32, 196), (6272, 196, 1), 0), out=buf490)
    buf491 = buf466; del buf466  # reuse
    buf492 = reinterpret_tensor(buf490, (8, 12, 196, 196), (460992, 38416, 196, 1), 0); del buf490  # reuse
    buf493 = empty_strided((8, 12, 196, 1), (2352, 196, 1, 18816), device='cpu', dtype=torch.float32)
    buf494 = empty((8, 12, 196, 196), device='cpu', dtype=torch.float32)
    buf495 = empty((8, 12, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_mul_113(c_void_p(buf492.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(buf495.data_ptr()))
    del buf491
    buf496 = empty((96, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf494, (96, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf495, (96, 196, 32), (6272, 32, 1), 0), out=buf496)
    buf497 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_view_114(c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()))
    buf498 = reinterpret_tensor(buf496, (1568, 384), (384, 1), 0); del buf496  # reuse
    # Source Nodes: [x_204], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_215, buf497, reinterpret_tensor(primals_214, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf498)
    del primals_215
    buf499 = reinterpret_tensor(buf498, (8, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf498  # reuse
    buf500 = buf482; del buf482  # reuse
    buf501 = empty_strided((8, 14, 14, 1), (196, 14, 1, 1568), device='cpu', dtype=torch.float32)
    buf503 = empty_strided((8, 14, 14, 384), (75264, 1, 5376, 14), device='cpu', dtype=torch.float32)
    buf504 = empty((1568, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_115(c_void_p(buf499.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()))
    del buf448
    del buf456
    del buf473
    del primals_217
    buf505 = buf487; del buf487  # reuse
    # Source Nodes: [x_207], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_219, buf504, reinterpret_tensor(primals_218, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf505)
    del primals_219
    buf506 = empty((1568, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_116(c_void_p(buf505.data_ptr()), c_void_p(buf506.data_ptr()))
    buf507 = buf481; del buf481  # reuse
    # Source Nodes: [x_211], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_221, buf506, reinterpret_tensor(primals_220, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf507)
    del primals_221
    buf508 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf509 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf510 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf512 = reinterpret_tensor(buf510, (8, 197, 1), (197, 1, 1), 0); del buf510  # reuse
    buf513 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_117(c_void_p(buf512.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf513.data_ptr()))
    del buf499
    del primals_2
    del primals_223
    buf514 = empty((1576, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___post_network_0_attn_kv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf513, (1576, 384), (384, 1), 0), reinterpret_tensor(primals_224, (384, 768), (1, 384), 0), out=buf514)
    buf515 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___post_network_0_attn_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf513, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_225, (384, 384), (1, 384), 0), out=buf515)
    buf516 = reinterpret_tensor(buf515, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf515  # reuse
    buf517 = empty((8, 12, 32, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_118(c_void_p(buf516.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf517.data_ptr()))
    buf518 = empty((96, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_62], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf516, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf517, (96, 32, 197), (6304, 197, 1), 0), out=buf518)
    buf519 = empty_strided((8, 12, 1, 1), (12, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf520 = reinterpret_tensor(buf518, (8, 12, 1, 197), (2364, 197, 18912, 1), 0); del buf518  # reuse
    buf521 = empty_strided((8, 12, 1, 1), (12, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf522 = empty((8, 12, 1, 197), device='cpu', dtype=torch.float32)
    buf523 = empty((8, 12, 197, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_119(c_void_p(buf520.data_ptr()), c_void_p(buf514.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()))
    buf524 = empty((96, 1, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf522, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf523, (96, 197, 32), (6304, 32, 1), 0), out=buf524)
    buf525 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_embed_2], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_227, reinterpret_tensor(buf524, (8, 384), (384, 1), 0), reinterpret_tensor(primals_226, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf525)
    del primals_227
    buf526 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf527 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf529 = empty((8, 1, 384), device='cpu', dtype=torch.float32)
    buf530 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_120(c_void_p(buf508.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf527.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(buf530.data_ptr()))
    del primals_229
    buf531 = empty((8, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_218], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_231, buf530, reinterpret_tensor(primals_230, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf531)
    del primals_231
    buf532 = empty((8, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_121(c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()))
    buf533 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_222], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_233, buf532, reinterpret_tensor(primals_232, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf533)
    del primals_233
    buf534 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf535 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf536 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf538 = reinterpret_tensor(buf536, (8, 197, 1), (197, 1, 1), 0); del buf536  # reuse
    buf539 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_122(c_void_p(buf538.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf533.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf539.data_ptr()))
    del primals_235
    buf540 = buf514; del buf514  # reuse
    # Source Nodes: [l__mod___post_network_1_attn_kv], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf539, (1576, 384), (384, 1), 0), reinterpret_tensor(primals_236, (384, 768), (1, 384), 0), out=buf540)
    buf541 = buf533; del buf533  # reuse
    # Source Nodes: [l__mod___post_network_1_attn_q], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf539, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_237, (384, 384), (1, 384), 0), out=buf541)
    buf542 = reinterpret_tensor(buf541, (8, 12, 1, 32), (384, 32, 32, 1), 0); del buf541  # reuse
    buf543 = empty((8, 12, 32, 197), device='cpu', dtype=torch.float32)
    cpp_fused_clone_mul_123(c_void_p(buf542.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf543.data_ptr()))
    buf544 = empty((96, 1, 197), device='cpu', dtype=torch.float32)
    # Source Nodes: [attn_65], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf542, (96, 1, 32), (32, 0, 1), 0), reinterpret_tensor(buf543, (96, 32, 197), (6304, 197, 1), 0), out=buf544)
    buf545 = buf519; del buf519  # reuse
    buf546 = reinterpret_tensor(buf544, (8, 12, 1, 197), (2364, 197, 18912, 1), 0); del buf544  # reuse
    buf547 = empty_strided((8, 12, 1, 1), (12, 1, 96, 96), device='cpu', dtype=torch.float32)
    buf548 = empty((8, 12, 1, 197), device='cpu', dtype=torch.float32)
    buf549 = empty((8, 12, 197, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_clone_124(c_void_p(buf546.data_ptr()), c_void_p(buf540.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf548.data_ptr()), c_void_p(buf549.data_ptr()))
    del buf540
    del buf545
    buf550 = reinterpret_tensor(buf525, (96, 1, 32), (32, 32, 1), 0); del buf525  # reuse
    # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf548, (96, 1, 197), (197, 0, 1), 0), reinterpret_tensor(buf549, (96, 197, 32), (6304, 32, 1), 0), out=buf550)
    buf551 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [cls_embed_8], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_239, reinterpret_tensor(buf550, (8, 384), (384, 1), 0), reinterpret_tensor(primals_238, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf551)
    del primals_239
    buf552 = buf526; del buf526  # reuse
    buf553 = empty_strided((8, 1, 1), (1, 8, 8), device='cpu', dtype=torch.float32)
    buf555 = empty((8, 1, 384), device='cpu', dtype=torch.float32)
    buf556 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused_add_native_layer_norm_view_125(c_void_p(buf534.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()))
    del buf552
    del primals_241
    buf557 = empty((8, 1152), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_225], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_243, buf556, reinterpret_tensor(primals_242, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf557)
    del primals_243
    buf558 = empty((8, 1152), device='cpu', dtype=torch.float32)
    cpp_fused_gelu_view_126(c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()))
    buf559 = empty((8, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_229], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_245, buf558, reinterpret_tensor(primals_244, (1152, 384), (1, 1152), 0), alpha=1, beta=1, out=buf559)
    del primals_245
    buf560 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    buf561 = empty((8, 197, 1), device='cpu', dtype=torch.float32)
    buf562 = empty_strided((8, 197, 1), (197, 1, 1576), device='cpu', dtype=torch.float32)
    buf564 = reinterpret_tensor(buf562, (8, 197, 1), (197, 1, 1), 0); del buf562  # reuse
    buf565 = empty((8, 197, 384), device='cpu', dtype=torch.float32)
    cpp_fused_cat_native_layer_norm_127(c_void_p(buf564.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf551.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf565.data_ptr()))
    del buf551
    del buf559
    del primals_247
    buf566 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [out_1], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_249, reinterpret_tensor(buf565, (8, 384), (75648, 1), 0), reinterpret_tensor(primals_248, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf566)
    del primals_249
    buf567 = buf507; del buf507  # reuse
    cpp_fused__unsafe_view_clone_128(c_void_p(buf565.data_ptr()), c_void_p(buf567.data_ptr()))
    buf568 = empty((1568, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [aux], Original ATen: [aten.mm]
    extern_kernels.mm(buf567, reinterpret_tensor(primals_250, (384, 1000), (1, 384), 0), out=buf568)
    buf569 = empty((8, 1000), device='cpu', dtype=torch.float32)
    buf570 = empty((8, 1000), device='cpu', dtype=torch.int64)
    buf571 = buf566; del buf566  # reuse
    buf572 = reinterpret_tensor(buf553, (8, 1, 1), (1, 1, 1), 0); del buf553  # reuse
    buf573 = empty_strided((8, 12, 1, 197), (2364, 1, 2364, 12), device='cpu', dtype=torch.float32)
    buf574 = reinterpret_tensor(buf527, (8, 1, 1), (1, 1, 1), 0); del buf527  # reuse
    buf575 = empty_strided((8, 12, 1, 197), (2364, 1, 2364, 12), device='cpu', dtype=torch.float32)
    buf576 = reinterpret_tensor(buf500, (8, 14, 14, 1), (196, 1, 14, 14), 0); del buf500  # reuse
    buf577 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf578 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf579 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf580 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf581 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf582 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf583 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf584 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf585 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf586 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf587 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf588 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf589 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf590 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf591 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf592 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf593 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf594 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf595 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf596 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf597 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf598 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf599 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf600 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf601 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf602 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf603 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf604 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf605 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf606 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf607 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf608 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf609 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf610 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf611 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf612 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf613 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf614 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf615 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf616 = empty_strided((8, 12, 196, 196), (460992, 1, 2352, 12), device='cpu', dtype=torch.float32)
    buf617 = empty_strided((8, 14, 14, 1), (196, 1, 14, 14), device='cpu', dtype=torch.float32)
    buf618 = reinterpret_tensor(buf141, (8, 28, 28, 1), (784, 1, 28, 28), 0); del buf141  # reuse
    buf619 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf620 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf621 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf622 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf623 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf624 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf625 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf626 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf627 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf628 = empty((8, 6, 196, 9, 9), device='cpu', dtype=torch.float32)
    buf629 = empty_strided((8, 28, 28, 1), (784, 1, 28, 28), device='cpu', dtype=torch.float32)
    buf634 = reinterpret_tensor(buf8, (64, ), (1, ), 0); del buf8  # reuse
    buf642 = reinterpret_tensor(buf14, (64, ), (1, ), 0); del buf14  # reuse
    buf650 = reinterpret_tensor(buf20, (64, ), (1, ), 0); del buf20  # reuse
    cpp_fused__native_batch_norm_legit_functional__softmax_add_detach_max_mul_native_layer_norm_native_layer_norm_backward_129(c_void_p(buf571.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf574.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf546.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf520.data_ptr()), c_void_p(buf521.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf450.data_ptr()), c_void_p(buf441.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf390.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf314.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf569.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf573.data_ptr()), c_void_p(buf575.data_ptr()), c_void_p(buf576.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf579.data_ptr()), c_void_p(buf580.data_ptr()), c_void_p(buf581.data_ptr()), c_void_p(buf582.data_ptr()), c_void_p(buf583.data_ptr()), c_void_p(buf584.data_ptr()), c_void_p(buf585.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(buf588.data_ptr()), c_void_p(buf589.data_ptr()), c_void_p(buf590.data_ptr()), c_void_p(buf591.data_ptr()), c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf595.data_ptr()), c_void_p(buf596.data_ptr()), c_void_p(buf597.data_ptr()), c_void_p(buf598.data_ptr()), c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(buf601.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf603.data_ptr()), c_void_p(buf604.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf606.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf609.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf611.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf613.data_ptr()), c_void_p(buf614.data_ptr()), c_void_p(buf615.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(buf617.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf619.data_ptr()), c_void_p(buf620.data_ptr()), c_void_p(buf621.data_ptr()), c_void_p(buf622.data_ptr()), c_void_p(buf623.data_ptr()), c_void_p(buf624.data_ptr()), c_void_p(buf625.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf627.data_ptr()), c_void_p(buf628.data_ptr()), c_void_p(buf629.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()))
    del buf111
    del buf120
    del buf129
    del buf130
    del buf142
    del buf152
    del buf161
    del buf162
    del buf169
    del buf177
    del buf186
    del buf187
    del buf195
    del buf203
    del buf212
    del buf213
    del buf220
    del buf228
    del buf237
    del buf238
    del buf246
    del buf254
    del buf26
    del buf263
    del buf264
    del buf271
    del buf279
    del buf288
    del buf289
    del buf297
    del buf305
    del buf314
    del buf315
    del buf322
    del buf330
    del buf339
    del buf340
    del buf348
    del buf356
    del buf36
    del buf365
    del buf366
    del buf37
    del buf373
    del buf381
    del buf390
    del buf391
    del buf399
    del buf407
    del buf416
    del buf417
    del buf424
    del buf432
    del buf441
    del buf442
    del buf450
    del buf458
    del buf467
    del buf468
    del buf475
    del buf483
    del buf492
    del buf493
    del buf50
    del buf501
    del buf520
    del buf521
    del buf546
    del buf547
    del buf568
    del buf569
    del buf58
    del buf634
    del buf642
    del buf650
    del buf67
    del buf68
    del buf81
    del buf89
    del buf98
    del buf99
    del primals_251
    del primals_252
    del primals_253
    del primals_254
    del primals_255
    del primals_256
    del primals_257
    del primals_258
    del primals_259
    del primals_260
    return (buf571, buf0, primals_4, buf1, primals_7, buf2, primals_10, buf3, primals_14, primals_21, primals_27, primals_34, primals_40, primals_47, primals_53, primals_60, buf4, primals_68, primals_73, primals_79, primals_84, primals_90, primals_95, primals_101, primals_106, primals_112, primals_117, primals_123, primals_128, primals_134, primals_139, primals_145, primals_150, primals_156, primals_161, primals_167, primals_172, primals_178, primals_183, primals_189, primals_194, primals_200, primals_205, primals_211, primals_216, primals_222, primals_228, primals_234, primals_240, primals_246, buf5, buf6, buf10, buf11, buf12, buf16, buf17, buf18, buf22, buf23, buf28, buf29, buf31, reinterpret_tensor(buf31, (3, 14, 1, 1), (14, 1, 1, 1), 0), buf32, buf33, buf41, buf47, buf52, buf53, buf54, buf55, buf60, buf61, buf63, buf64, buf77, buf83, buf84, buf85, buf86, buf91, buf92, buf94, buf95, buf108, buf113, buf114, buf115, buf116, buf122, buf123, buf125, buf126, buf139, buf144, buf145, buf146, buf147, buf149, buf154, buf155, buf166, buf171, buf172, buf173, buf174, buf179, buf180, buf191, buf197, buf198, buf199, buf200, buf205, buf206, buf217, buf222, buf223, buf224, buf225, buf230, buf231, buf242, buf248, buf249, buf250, buf251, buf256, buf257, buf268, buf273, buf274, buf275, buf276, buf281, buf282, buf293, buf299, buf300, buf301, buf302, buf307, buf308, buf319, buf324, buf325, buf326, buf327, buf332, buf333, buf344, buf350, buf351, buf352, buf353, buf358, buf359, buf370, buf375, buf376, buf377, buf378, buf383, buf384, buf395, buf401, buf402, buf403, buf404, buf409, buf410, buf421, buf426, buf427, buf428, buf429, buf434, buf435, buf446, buf452, buf453, buf454, buf455, buf460, buf461, buf472, buf477, buf478, buf479, buf480, buf485, buf486, buf497, buf503, buf504, buf505, buf506, buf508, buf509, buf512, reinterpret_tensor(buf513, (1576, 384), (384, 1), 0), reinterpret_tensor(buf513, (8, 384), (75648, 1), 0), reinterpret_tensor(buf524, (8, 384), (384, 1), 0), buf529, buf530, buf531, buf532, buf534, buf535, buf538, reinterpret_tensor(buf539, (1576, 384), (384, 1), 0), reinterpret_tensor(buf539, (8, 384), (75648, 1), 0), reinterpret_tensor(buf550, (8, 384), (384, 1), 0), buf555, buf556, buf557, buf558, buf560, buf561, buf564, reinterpret_tensor(buf565, (8, 384), (75648, 1), 0), buf567, reinterpret_tensor(buf570, (8, 1, 1000), (1000, 1000, 1), 0), reinterpret_tensor(primals_250, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_248, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_244, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_242, (1152, 384), (384, 1), 0), buf572, reinterpret_tensor(primals_238, (384, 384), (384, 1), 0), reinterpret_tensor(buf548, (96, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf549, (96, 32, 197), (6304, 1, 32), 0), buf573, reinterpret_tensor(buf542, (96, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf543, (96, 197, 32), (6304, 1, 197), 0), reinterpret_tensor(primals_237, (384, 384), (384, 1), 0), reinterpret_tensor(primals_236, (768, 384), (384, 1), 0), reinterpret_tensor(primals_232, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_230, (1152, 384), (384, 1), 0), buf574, reinterpret_tensor(primals_226, (384, 384), (384, 1), 0), reinterpret_tensor(buf522, (96, 197, 1), (197, 1, 0), 0), reinterpret_tensor(buf523, (96, 32, 197), (6304, 1, 32), 0), buf575, reinterpret_tensor(buf516, (96, 32, 1), (32, 1, 0), 0), reinterpret_tensor(buf517, (96, 197, 32), (6304, 1, 197), 0), reinterpret_tensor(primals_225, (384, 384), (384, 1), 0), reinterpret_tensor(primals_224, (768, 384), (384, 1), 0), reinterpret_tensor(primals_220, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_218, (1152, 384), (384, 1), 0), buf576, reinterpret_tensor(primals_214, (384, 384), (384, 1), 0), reinterpret_tensor(buf494, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf495, (96, 32, 196), (6272, 1, 32), 0), buf577, reinterpret_tensor(buf488, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf489, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_213, (1152, 384), (384, 1), 0), buf578, reinterpret_tensor(primals_209, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_207, (1152, 384), (384, 1), 0), buf579, reinterpret_tensor(primals_203, (384, 384), (384, 1), 0), reinterpret_tensor(buf469, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf470, (96, 32, 196), (6272, 1, 32), 0), buf580, reinterpret_tensor(buf463, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf464, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_202, (1152, 384), (384, 1), 0), buf581, reinterpret_tensor(primals_198, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_196, (1152, 384), (384, 1), 0), buf582, reinterpret_tensor(primals_192, (384, 384), (384, 1), 0), reinterpret_tensor(buf443, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf444, (96, 32, 196), (6272, 1, 32), 0), buf583, reinterpret_tensor(buf437, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf438, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_191, (1152, 384), (384, 1), 0), buf584, reinterpret_tensor(primals_187, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_185, (1152, 384), (384, 1), 0), buf585, reinterpret_tensor(primals_181, (384, 384), (384, 1), 0), reinterpret_tensor(buf418, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf419, (96, 32, 196), (6272, 1, 32), 0), buf586, reinterpret_tensor(buf412, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf413, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_180, (1152, 384), (384, 1), 0), buf587, reinterpret_tensor(primals_176, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_174, (1152, 384), (384, 1), 0), buf588, reinterpret_tensor(primals_170, (384, 384), (384, 1), 0), reinterpret_tensor(buf392, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf393, (96, 32, 196), (6272, 1, 32), 0), buf589, reinterpret_tensor(buf386, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf387, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_169, (1152, 384), (384, 1), 0), buf590, reinterpret_tensor(primals_165, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_163, (1152, 384), (384, 1), 0), buf591, reinterpret_tensor(primals_159, (384, 384), (384, 1), 0), reinterpret_tensor(buf367, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf368, (96, 32, 196), (6272, 1, 32), 0), buf592, reinterpret_tensor(buf361, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf362, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_158, (1152, 384), (384, 1), 0), buf593, reinterpret_tensor(primals_154, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_152, (1152, 384), (384, 1), 0), buf594, reinterpret_tensor(primals_148, (384, 384), (384, 1), 0), reinterpret_tensor(buf341, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf342, (96, 32, 196), (6272, 1, 32), 0), buf595, reinterpret_tensor(buf335, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf336, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_147, (1152, 384), (384, 1), 0), buf596, reinterpret_tensor(primals_143, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_141, (1152, 384), (384, 1), 0), buf597, reinterpret_tensor(primals_137, (384, 384), (384, 1), 0), reinterpret_tensor(buf316, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf317, (96, 32, 196), (6272, 1, 32), 0), buf598, reinterpret_tensor(buf310, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf311, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_136, (1152, 384), (384, 1), 0), buf599, reinterpret_tensor(primals_132, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_130, (1152, 384), (384, 1), 0), buf600, reinterpret_tensor(primals_126, (384, 384), (384, 1), 0), reinterpret_tensor(buf290, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf291, (96, 32, 196), (6272, 1, 32), 0), buf601, reinterpret_tensor(buf284, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf285, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_125, (1152, 384), (384, 1), 0), buf602, reinterpret_tensor(primals_121, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_119, (1152, 384), (384, 1), 0), buf603, reinterpret_tensor(primals_115, (384, 384), (384, 1), 0), reinterpret_tensor(buf265, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf266, (96, 32, 196), (6272, 1, 32), 0), buf604, reinterpret_tensor(buf259, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf260, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_114, (1152, 384), (384, 1), 0), buf605, reinterpret_tensor(primals_110, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_108, (1152, 384), (384, 1), 0), buf606, reinterpret_tensor(primals_104, (384, 384), (384, 1), 0), reinterpret_tensor(buf239, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf240, (96, 32, 196), (6272, 1, 32), 0), buf607, reinterpret_tensor(buf233, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf234, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_103, (1152, 384), (384, 1), 0), buf608, reinterpret_tensor(primals_99, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_97, (1152, 384), (384, 1), 0), buf609, reinterpret_tensor(primals_93, (384, 384), (384, 1), 0), reinterpret_tensor(buf214, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf215, (96, 32, 196), (6272, 1, 32), 0), buf610, reinterpret_tensor(buf208, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf209, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_92, (1152, 384), (384, 1), 0), buf611, reinterpret_tensor(primals_88, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_86, (1152, 384), (384, 1), 0), buf612, reinterpret_tensor(primals_82, (384, 384), (384, 1), 0), reinterpret_tensor(buf188, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf189, (96, 32, 196), (6272, 1, 32), 0), buf613, reinterpret_tensor(buf182, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf183, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_81, (1152, 384), (384, 1), 0), buf614, reinterpret_tensor(primals_77, (384, 1152), (1152, 1), 0), reinterpret_tensor(primals_75, (1152, 384), (384, 1), 0), buf615, reinterpret_tensor(primals_71, (384, 384), (384, 1), 0), reinterpret_tensor(buf163, (96, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf164, (96, 32, 196), (6272, 1, 32), 0), buf616, reinterpret_tensor(buf157, (96, 32, 196), (6272, 1, 32), 0), reinterpret_tensor(buf158, (96, 196, 32), (6272, 1, 196), 0), reinterpret_tensor(primals_70, (1152, 384), (384, 1), 0), buf617, reinterpret_tensor(primals_64, (192, 576), (576, 1), 0), reinterpret_tensor(primals_62, (576, 192), (192, 1), 0), buf618, reinterpret_tensor(primals_58, (192, 192), (192, 1), 0), reinterpret_tensor(buf131, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf132, (9408, 32, 9), (288, 1, 32), 0), buf619, reinterpret_tensor(primals_56, (486, 192), (192, 1), 0), reinterpret_tensor(primals_55, (192, 192), (192, 1), 0), buf620, reinterpret_tensor(primals_51, (192, 576), (576, 1), 0), reinterpret_tensor(primals_49, (576, 192), (192, 1), 0), buf621, reinterpret_tensor(primals_45, (192, 192), (192, 1), 0), reinterpret_tensor(buf100, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf101, (9408, 32, 9), (288, 1, 32), 0), buf622, reinterpret_tensor(primals_43, (486, 192), (192, 1), 0), reinterpret_tensor(primals_42, (192, 192), (192, 1), 0), buf623, reinterpret_tensor(primals_38, (192, 576), (576, 1), 0), reinterpret_tensor(primals_36, (576, 192), (192, 1), 0), buf624, reinterpret_tensor(primals_32, (192, 192), (192, 1), 0), reinterpret_tensor(buf69, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf70, (9408, 32, 9), (288, 1, 32), 0), buf625, reinterpret_tensor(primals_30, (486, 192), (192, 1), 0), reinterpret_tensor(primals_29, (192, 192), (192, 1), 0), buf626, reinterpret_tensor(primals_25, (192, 576), (576, 1), 0), reinterpret_tensor(primals_23, (576, 192), (192, 1), 0), buf627, reinterpret_tensor(primals_19, (192, 192), (192, 1), 0), reinterpret_tensor(buf38, (9408, 9, 9), (81, 1, 9), 0), reinterpret_tensor(buf39, (9408, 32, 9), (288, 1, 32), 0), buf628, reinterpret_tensor(primals_17, (486, 192), (192, 1), 0), reinterpret_tensor(primals_16, (192, 192), (192, 1), 0), buf629, reinterpret_tensor(buf19, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf7, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 14, 14, 384), (75264, 5376, 384, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((1, 1, 384), (384, 384, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((192, 64, 4, 4), (1024, 16, 4, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((486, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((486, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((192, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((576, 192), (192, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((576, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((192, 576), (576, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((384, 192, 2, 2), (768, 4, 2, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_216 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((1152, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((1152, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((384, 1152), (1152, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_255 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_258 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_261 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('volo_d1_224', benchmark_compiled_module)
