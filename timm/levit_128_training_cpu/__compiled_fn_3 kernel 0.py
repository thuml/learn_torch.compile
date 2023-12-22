
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr2[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr2 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr4[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr4[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(100352L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(25088L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(25088.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(6272L); x1+=static_cast<long>(1L))
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
                    auto tmp1 = static_cast<float>(6272.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(6272.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_clone_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                tmp0.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)), static_cast<long>(256L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(1568.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 196);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 196);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)), static_cast<long>(256L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(1568.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 196);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 196);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)), static_cast<long>(256L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(1568.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 196);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 196);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_19 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)), static_cast<long>(256L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(1568.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(1568.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 196);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))] = tmp_acc0;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (196L*x1) + (784L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 196);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (38416L*x1) + (153664L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (256L*x2) + (50176L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_25 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(1568.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1568.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp7 + tmp9;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp12 = tmp3 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp16 = tmp14 + tmp15;
                    auto tmp17 = tmp0 + tmp16;
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_view_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(1568L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(49L))) % static_cast<long>(7L))) + (3584L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(49L)), 7L))) + (25088L*(c10::div_floor_integer(x0, 49L)))));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (6272L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(392.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
                    }
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (80L*x1) + (640L*x3) + (125440L*x0)), static_cast<long>(640L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = in_ptr4[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp5 = in_ptr5[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp13 = in_ptr6[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp16 = in_ptr7[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = static_cast<float>(1568.0);
                                auto tmp7 = tmp5 / tmp6;
                                auto tmp8 = static_cast<float>(1e-05);
                                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                auto tmp10 = 1 / std::sqrt(tmp9);
                                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                auto tmp12 = tmp4 * tmp11;
                                auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                auto tmp15 = tmp12 * tmp14;
                                auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                auto tmp18 = tmp15 + tmp17;
                                tmp18.store(out_ptr4 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (25088L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (80L*x1) + (640L*x3) + (125440L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (25088L*x0))] = tmpbuf[x2_inner]; }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 196);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (196L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 196);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 196L), "index out of bounds: 0 <= tmp6 < 196L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (196L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (196L*x2) + (9604L*x1) + (76832L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x3 + (80L*x1) + (640L*x2) + (125440L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(1568.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (3136L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(392.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_35 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)), static_cast<long>(512L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(392.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 49);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 49);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (1568L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (12544L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_39 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_41 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)), static_cast<long>(512L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(392.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 49);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 49);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (1568L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (12544L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_45 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_47 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)), static_cast<long>(512L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(392.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 49);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 49);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (1568L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (12544L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_51 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_53 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_54 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                        {
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                            {
                                float tmp0[8*8] __attribute__ ((aligned (8)));
                                at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)), static_cast<long>(512L), tmp0, 8);
                                for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                                {
                                    auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                    auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                    auto tmp4 = tmp1 - tmp3;
                                    auto tmp6 = static_cast<float>(392.0);
                                    auto tmp7 = tmp5 / tmp6;
                                    auto tmp8 = static_cast<float>(1e-05);
                                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                    auto tmp10 = 1 / std::sqrt(tmp9);
                                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                                    auto tmp12 = tmp4 * tmp11;
                                    auto tmp14 = at::vec::Vectorized<float>(tmp13);
                                    auto tmp15 = tmp12 * tmp14;
                                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                                    auto tmp18 = tmp15 + tmp17;
                                    tmp18.store(out_ptr4 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                                }
                            }
                            #pragma GCC ivdep
                            for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        {
                            float tmp_acc0 = -std::numeric_limits<float>::infinity();
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                            {
                                auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                                auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                                auto tmp1 = static_cast<float>(0.25);
                                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                                auto tmp4 = decltype(tmp3)(tmp3 + 49);
                                auto tmp5 = tmp3 < 0;
                                auto tmp6 = tmp5 ? tmp4 : tmp3;
                                TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                                auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                                auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                                tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                            }
                            out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))] = tmp_acc0;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp9 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (392L*x0))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 49);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                            auto tmp11 = std::exp(tmp10);
                            in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (2401L*x1) + (19208L*x0))] = tmp11;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                    for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                        tmp_acc0 = tmp_acc0 + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x0)];
                    auto tmp2 = tmp0 / tmp1;
                    in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (512L*x2) + (25088L*x0)));
                                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp2 = tmp0 - tmp1;
                                auto tmp4 = static_cast<float>(392.0);
                                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                                auto tmp6 = tmp3 / tmp5;
                                auto tmp7 = static_cast<float>(1e-05);
                                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                                auto tmp9 = tmp6 + tmp8;
                                auto tmp10 = tmp9.rsqrt();
                                auto tmp11 = tmp2 * tmp10;
                                auto tmp13 = tmp11 * tmp12;
                                auto tmp15 = tmp13 + tmp14;
                                tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (1568L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (12544L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_57 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (512L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(392.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 / tmp5;
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp11 = tmp2 * tmp10;
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp15 = tmp13 + tmp14;
                    tmp15.store(out_ptr3 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr4 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_59 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(392.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(392.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_view_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                    #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                    Welford<float> tmp_acc0 = Welford<float>();
                    Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1280L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
                {
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (512L*(static_cast<long>((static_cast<long>(x0) % static_cast<long>(16L))) % static_cast<long>(4L))) + (3584L*(c10::div_floor_integer((static_cast<long>(x0) % static_cast<long>(16L)), 4L))) + (12544L*(c10::div_floor_integer(x0, 16L)))));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (256L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (4096L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(48L); x3+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr3 + static_cast<long>(x2 + (80L*x1) + (1280L*x3) + (62720L*x0)), static_cast<long>(1280L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = in_ptr4[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp5 = in_ptr5[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp13 = in_ptr6[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp16 = in_ptr7[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(392.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr4 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (80L*x1) + (1280L*x3) + (62720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(392.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp15.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr4[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 49);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                        }
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x3 + (49L*x2))];
                        auto tmp9 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (256L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp3)(tmp3 + 49);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 49L), "index out of bounds: 0 <= tmp6 < 49L")
                        auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (49L*x1))];
                        auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = std::exp(tmp10);
                        in_out_ptr0[static_cast<long>(x3 + (49L*x2) + (784L*x1) + (12544L*x0))] = tmp11;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                #pragma omp simd simdlen(4)  reduction(+:tmp_acc0)
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    tmp_acc0 = tmp_acc0 + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = tmp0 / tmp1;
                in_out_ptr0[static_cast<long>(x1 + (49L*x0))] = tmp2;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x3 + (80L*x1) + (1280L*x2) + (62720L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp2 = tmp0 - tmp1;
                            auto tmp4 = static_cast<float>(392.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = tmp3 / tmp5;
                            auto tmp7 = static_cast<float>(1e-05);
                            auto tmp8 = at::vec::Vectorized<float>(tmp7);
                            auto tmp9 = tmp6 + tmp8;
                            auto tmp10 = tmp9.rsqrt();
                            auto tmp11 = tmp2 * tmp10;
                            auto tmp13 = tmp11 * tmp12;
                            auto tmp15 = tmp13 + tmp14;
                            tmp15.store(out_ptr2 + static_cast<long>(x3 + (64L*x2) + (3136L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (1024L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (16384L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(3.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 + tmp2;
                auto tmp4 = static_cast<float>(0.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = at::vec::maximum(tmp3, tmp5);
                auto tmp7 = static_cast<float>(6.0);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = at::vec::minimum(tmp6, tmp8);
                auto tmp10 = tmp0 * tmp9;
                auto tmp11 = tmp10 / tmp8;
                tmp11.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_65 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_66 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_67 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (768L*x3) + (12288L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(128.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 16);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                        }
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                        auto tmp9 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp3)(tmp3 + 16);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                        auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                        auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = std::exp(tmp10);
                        in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))] = tmp11;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (512L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (6144L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_70 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_71 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_72 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_73 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (768L*x3) + (12288L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(128.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 16);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                        }
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                        auto tmp9 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp3)(tmp3 + 16);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                        auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                        auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = std::exp(tmp10);
                        in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))] = tmp11;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (512L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (6144L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_76 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_77 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_78 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_79 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (768L*x3) + (12288L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(128.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 16);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                        }
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                        auto tmp9 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp3)(tmp3 + 16);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                        auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                        auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = std::exp(tmp10);
                        in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))] = tmp11;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (512L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (6144L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_82 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_83 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_84 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_clone_85 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr3 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (768L*x3) + (12288L*x0)), static_cast<long>(768L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = out_ptr0[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = out_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp13 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp16 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(128.0);
                            auto tmp7 = tmp5 / tmp6;
                            auto tmp8 = static_cast<float>(1e-05);
                            auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                            auto tmp10 = 1 / std::sqrt(tmp9);
                            auto tmp11 = at::vec::Vectorized<float>(tmp10);
                            auto tmp12 = tmp4 * tmp11;
                            auto tmp14 = at::vec::Vectorized<float>(tmp13);
                            auto tmp15 = tmp12 * tmp14;
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            tmp18.store(out_ptr4 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__softmax_add_clone_index_mul_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const long* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    {
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                            auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp3)(tmp3 + 16);
                            auto tmp5 = tmp3 < 0;
                            auto tmp6 = tmp5 ? tmp4 : tmp3;
                            TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                            auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                            auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp8);
                        }
                        out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))] = tmp_acc0;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))];
                        auto tmp3 = in_ptr1[static_cast<long>(x3 + (16L*x2))];
                        auto tmp9 = out_ptr0[static_cast<long>(x2 + (16L*x1) + (192L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp3)(tmp3 + 16);
                        auto tmp5 = tmp3 < 0;
                        auto tmp6 = tmp5 ? tmp4 : tmp3;
                        TORCH_CHECK((0 <= tmp6) & (tmp6 < 16L), "index out of bounds: 0 <= tmp6 < 16L")
                        auto tmp7 = in_ptr2[static_cast<long>(tmp6 + (16L*x1))];
                        auto tmp8 = decltype(tmp2)(tmp2 + tmp7);
                        auto tmp10 = decltype(tmp8)(tmp8 - tmp9);
                        auto tmp11 = std::exp(tmp10);
                        in_out_ptr0[static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0))] = tmp11;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr1[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                auto tmp1 = out_ptr1[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(32L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(32L + x3 + (64L*x1) + (768L*x2) + (12288L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(128.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = tmp3 / tmp5;
                        auto tmp7 = static_cast<float>(1e-05);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 + tmp8;
                        auto tmp10 = tmp9.rsqrt();
                        auto tmp11 = tmp2 * tmp10;
                        auto tmp13 = tmp11 * tmp12;
                        auto tmp15 = tmp13 + tmp14;
                        tmp15.store(out_ptr2 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__unsafe_view_clone_hardswish_view_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(384L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (512L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (6144L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(49152L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr1 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_88 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(128.0);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 / tmp6;
                auto tmp8 = static_cast<float>(1e-05);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp12 = tmp3 * tmp11;
                auto tmp14 = tmp12 * tmp13;
                auto tmp16 = tmp14 + tmp15;
                auto tmp17 = tmp0 + tmp16;
                tmp17.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_hardswish_view_89 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(128.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                tmp15.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 + tmp2;
            auto tmp4 = static_cast<float>(0.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = at::vec::maximum(tmp3, tmp5);
            auto tmp7 = static_cast<float>(6.0);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = at::vec::minimum(tmp6, tmp8);
            auto tmp10 = tmp0 * tmp9;
            auto tmp11 = tmp10 / tmp8;
            tmp11.store(out_ptr4 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_mean_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr3 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr1 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1e-05);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 + tmp5;
            auto tmp7 = tmp6.rsqrt();
            tmp7.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x2) + (6144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (6144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(128.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 / tmp6;
                        auto tmp8 = static_cast<float>(1e-05);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 + tmp9;
                        auto tmp11 = tmp10.rsqrt();
                        auto tmp12 = tmp3 * tmp11;
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp16 = tmp14 + tmp15;
                        auto tmp17 = tmp0 + tmp16;
                        tmp_acc0_vec = tmp_acc0_vec + tmp17;
                    }
                    tmp_acc0_vec.store(out_ptr3 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3072L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr3 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(16.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            tmp3.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(welford:Welford<float>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<float>()})
                #pragma omp declare reduction(welford:Welford<at::vec::Vectorized<float>>:omp_out = welford_combine(omp_out, omp_in)) initializer(omp_priv={Welford<at::vec::Vectorized<float>>()})
                Welford<float> tmp_acc0 = Welford<float>();
                Welford<at::vec::Vectorized<float>> tmp_acc0_vec = Welford<at::vec::Vectorized<float>>();
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (384L*x1)));
                    tmp_acc0_vec = welford_combine(tmp_acc0_vec, tmp0);
                }
                tmp_acc0_vec.mean.store(out_ptr4 + static_cast<long>(x0));
                tmp_acc0_vec.m2.store(out_ptr5 + static_cast<long>(x0));
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr4 + static_cast<long>(x1));
                auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr5 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp2 = tmp0 - tmp1;
                auto tmp4 = static_cast<float>(8.0);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 / tmp5;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.rsqrt();
                auto tmp11 = tmp2 * tmp10;
                auto tmp13 = tmp11 * tmp12;
                auto tmp15 = tmp13 + tmp14;
                auto tmp17 = tmp11 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                tmp15.store(out_ptr6 + static_cast<long>(x1 + (384L*x0)));
                tmp19.store(out_ptr7 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_detach_div_91 = async_compile.cpp('''
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
                       const long* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       const float* in_ptr21,
                       const long* in_ptr22,
                       const float* in_ptr23,
                       const float* in_ptr24,
                       const float* in_ptr25,
                       const long* in_ptr26,
                       const float* in_ptr27,
                       const float* in_ptr28,
                       const float* in_ptr29,
                       const long* in_ptr30,
                       const float* in_ptr31,
                       const float* in_ptr32,
                       const float* in_ptr33,
                       const long* in_ptr34,
                       const float* in_ptr35,
                       const float* in_ptr36,
                       const float* in_ptr37,
                       const long* in_ptr38,
                       const float* in_ptr39,
                       const float* in_ptr40,
                       const float* in_ptr41,
                       const long* in_ptr42,
                       const float* in_ptr43,
                       const float* in_ptr44,
                       const float* in_ptr45,
                       const long* in_ptr46,
                       const float* in_ptr47,
                       const float* in_ptr48,
                       const float* in_ptr49,
                       const long* in_ptr50,
                       const float* in_ptr51,
                       const float* in_ptr52,
                       const float* in_ptr53,
                       const long* in_ptr54,
                       const float* in_ptr55,
                       const float* in_ptr56,
                       const float* in_ptr57,
                       const long* in_ptr58,
                       const float* in_ptr59,
                       const float* in_ptr60,
                       const float* in_ptr61,
                       const long* in_ptr62,
                       const float* in_ptr63,
                       const float* in_ptr64,
                       const float* in_ptr65,
                       const long* in_ptr66,
                       const float* in_ptr67,
                       const float* in_ptr68,
                       const float* in_ptr69,
                       const long* in_ptr70,
                       const float* in_ptr71,
                       const float* in_ptr72,
                       const float* in_ptr73,
                       const long* in_ptr74,
                       const float* in_ptr75,
                       const float* in_ptr76,
                       const float* in_ptr77,
                       const long* in_ptr78,
                       const float* in_ptr79,
                       const float* in_ptr80,
                       const float* in_ptr81,
                       const long* in_ptr82,
                       const float* in_ptr83,
                       const float* in_ptr84,
                       const float* in_ptr85,
                       const long* in_ptr86,
                       const float* in_ptr87,
                       const float* in_ptr88,
                       const float* in_ptr89,
                       const long* in_ptr90,
                       const float* in_ptr91,
                       const float* in_ptr92,
                       const float* in_ptr93,
                       const long* in_ptr94,
                       const float* in_ptr95,
                       const float* in_ptr96,
                       const float* in_ptr97,
                       const long* in_ptr98,
                       const float* in_ptr99,
                       const float* in_ptr100,
                       const float* in_ptr101,
                       const long* in_ptr102,
                       const float* in_ptr103,
                       const float* in_ptr104,
                       const float* in_ptr105,
                       const long* in_ptr106,
                       const float* in_ptr107,
                       const float* in_ptr108,
                       const float* in_ptr109,
                       const long* in_ptr110,
                       const float* in_ptr111,
                       const float* in_ptr112,
                       const float* in_ptr113,
                       const long* in_ptr114,
                       const float* in_ptr115,
                       const float* in_ptr116,
                       const float* in_ptr117,
                       const long* in_ptr118,
                       const float* in_ptr119,
                       const float* in_ptr120,
                       const float* in_ptr121,
                       const long* in_ptr122,
                       const float* in_ptr123,
                       const float* in_ptr124,
                       const float* in_ptr125,
                       const long* in_ptr126,
                       const float* in_ptr127,
                       const float* in_ptr128,
                       const float* in_ptr129,
                       const long* in_ptr130,
                       const float* in_ptr131,
                       const float* in_ptr132,
                       const float* in_ptr133,
                       const long* in_ptr134,
                       const float* in_ptr135,
                       const float* in_ptr136,
                       const float* in_ptr137,
                       const long* in_ptr138,
                       const float* in_ptr139,
                       const float* in_ptr140,
                       const float* in_ptr141,
                       const long* in_ptr142,
                       const float* in_ptr143,
                       const float* in_ptr144,
                       const float* in_ptr145,
                       const long* in_ptr146,
                       const float* in_ptr147,
                       const float* in_ptr148,
                       const float* in_ptr149,
                       const long* in_ptr150,
                       const float* in_ptr151,
                       const float* in_ptr152,
                       const float* in_ptr153,
                       const long* in_ptr154,
                       const float* in_ptr155,
                       const float* in_ptr156,
                       const float* in_ptr157,
                       const long* in_ptr158,
                       const float* in_ptr159,
                       const float* in_ptr160,
                       const float* in_ptr161,
                       const long* in_ptr162,
                       const float* in_ptr163,
                       const float* in_ptr164,
                       const float* in_ptr165,
                       const long* in_ptr166,
                       const float* in_ptr167,
                       const float* in_ptr168,
                       const float* in_ptr169,
                       const long* in_ptr170,
                       const float* in_ptr171,
                       const float* in_ptr172,
                       const float* in_ptr173,
                       const long* in_ptr174,
                       const float* in_ptr175,
                       const float* in_ptr176,
                       const float* in_ptr177,
                       const long* in_ptr178,
                       const float* in_ptr179,
                       const float* in_ptr180,
                       const float* in_ptr181,
                       const long* in_ptr182,
                       const float* in_ptr183,
                       const float* in_ptr184,
                       const float* in_ptr185,
                       const long* in_ptr186,
                       const float* in_ptr187,
                       const float* in_ptr188,
                       const float* in_ptr189,
                       const long* in_ptr190,
                       const float* in_ptr191,
                       const float* in_ptr192,
                       const float* in_ptr193,
                       const long* in_ptr194,
                       const float* in_ptr195,
                       const float* in_ptr196,
                       const float* in_ptr197,
                       const long* in_ptr198,
                       const float* in_ptr199,
                       const float* in_ptr200,
                       const float* in_ptr201,
                       const long* in_ptr202,
                       const float* in_ptr203,
                       const float* in_ptr204,
                       const float* in_ptr205,
                       const long* in_ptr206,
                       const float* in_ptr207,
                       const float* in_ptr208,
                       const float* in_ptr209,
                       const long* in_ptr210,
                       const float* in_ptr211,
                       const float* in_ptr212,
                       const float* in_ptr213,
                       const long* in_ptr214,
                       const float* in_ptr215,
                       const float* in_ptr216,
                       const float* in_ptr217,
                       const long* in_ptr218,
                       const float* in_ptr219,
                       const float* in_ptr220,
                       const float* in_ptr221,
                       const long* in_ptr222,
                       const float* in_ptr223,
                       const float* in_ptr224,
                       const float* in_ptr225,
                       const long* in_ptr226,
                       const float* in_ptr227,
                       const float* in_ptr228,
                       const float* in_ptr229,
                       const long* in_ptr230,
                       const float* in_ptr231,
                       const float* in_ptr232,
                       const float* in_ptr233,
                       const long* in_ptr234,
                       const float* in_ptr235,
                       const float* in_ptr236,
                       const float* in_ptr237,
                       const long* in_ptr238,
                       const float* in_ptr239,
                       const float* in_ptr240,
                       const float* in_ptr241,
                       const long* in_ptr242,
                       const float* in_ptr243,
                       const float* in_ptr244,
                       const float* in_ptr245,
                       const long* in_ptr246,
                       const float* in_ptr247,
                       const float* in_ptr248,
                       const float* in_ptr249,
                       const long* in_ptr250,
                       float* out_ptr0,
                       float* out_ptr1,
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
                       float* out_ptr15,
                       float* out_ptr16,
                       long* out_ptr18,
                       float* out_ptr20,
                       float* out_ptr21,
                       long* out_ptr23,
                       float* out_ptr25,
                       float* out_ptr26,
                       long* out_ptr28,
                       float* out_ptr30,
                       float* out_ptr31,
                       long* out_ptr33,
                       float* out_ptr35,
                       float* out_ptr36,
                       long* out_ptr38,
                       float* out_ptr40,
                       float* out_ptr41,
                       long* out_ptr43,
                       float* out_ptr45,
                       float* out_ptr46,
                       long* out_ptr48,
                       float* out_ptr50,
                       float* out_ptr51,
                       long* out_ptr53,
                       float* out_ptr55,
                       float* out_ptr56,
                       long* out_ptr58,
                       float* out_ptr60,
                       float* out_ptr61,
                       long* out_ptr63,
                       float* out_ptr65,
                       float* out_ptr66,
                       long* out_ptr68,
                       float* out_ptr70,
                       float* out_ptr71,
                       long* out_ptr73,
                       float* out_ptr75,
                       float* out_ptr76,
                       long* out_ptr78,
                       float* out_ptr80,
                       float* out_ptr81,
                       long* out_ptr83,
                       float* out_ptr85,
                       float* out_ptr86,
                       long* out_ptr88,
                       float* out_ptr90,
                       float* out_ptr91,
                       long* out_ptr93,
                       float* out_ptr95,
                       float* out_ptr96,
                       long* out_ptr98,
                       float* out_ptr100,
                       float* out_ptr101,
                       long* out_ptr103,
                       float* out_ptr105,
                       float* out_ptr106,
                       long* out_ptr108,
                       float* out_ptr110,
                       float* out_ptr111,
                       long* out_ptr113,
                       float* out_ptr115,
                       float* out_ptr116,
                       long* out_ptr118,
                       float* out_ptr120,
                       float* out_ptr121,
                       long* out_ptr123,
                       float* out_ptr125,
                       float* out_ptr126,
                       long* out_ptr128,
                       float* out_ptr130,
                       float* out_ptr131,
                       long* out_ptr133,
                       float* out_ptr135,
                       float* out_ptr136,
                       long* out_ptr138,
                       float* out_ptr140,
                       float* out_ptr141,
                       long* out_ptr143,
                       float* out_ptr145,
                       float* out_ptr146,
                       long* out_ptr148,
                       float* out_ptr150,
                       float* out_ptr151,
                       long* out_ptr153,
                       float* out_ptr155,
                       float* out_ptr156,
                       long* out_ptr158,
                       float* out_ptr160,
                       float* out_ptr161,
                       long* out_ptr163,
                       float* out_ptr165,
                       float* out_ptr166,
                       long* out_ptr168,
                       float* out_ptr170,
                       float* out_ptr171,
                       long* out_ptr173,
                       float* out_ptr175,
                       float* out_ptr176,
                       long* out_ptr178,
                       float* out_ptr180,
                       float* out_ptr181,
                       long* out_ptr183,
                       float* out_ptr185,
                       float* out_ptr186,
                       long* out_ptr188,
                       float* out_ptr190,
                       float* out_ptr191,
                       long* out_ptr193,
                       float* out_ptr195,
                       float* out_ptr196,
                       long* out_ptr198,
                       float* out_ptr200,
                       float* out_ptr201,
                       long* out_ptr203,
                       float* out_ptr205,
                       float* out_ptr206,
                       long* out_ptr208,
                       float* out_ptr210,
                       float* out_ptr211,
                       long* out_ptr213,
                       float* out_ptr215,
                       float* out_ptr216,
                       long* out_ptr218,
                       float* out_ptr220,
                       float* out_ptr221,
                       long* out_ptr223,
                       float* out_ptr225,
                       float* out_ptr226,
                       long* out_ptr228,
                       float* out_ptr230,
                       float* out_ptr231,
                       long* out_ptr233,
                       float* out_ptr235,
                       float* out_ptr236,
                       long* out_ptr238,
                       float* out_ptr240,
                       float* out_ptr241,
                       long* out_ptr243,
                       float* out_ptr245,
                       float* out_ptr246,
                       long* out_ptr248,
                       float* out_ptr250,
                       float* out_ptr251,
                       long* out_ptr253,
                       float* out_ptr255,
                       float* out_ptr256,
                       long* out_ptr258,
                       float* out_ptr260,
                       float* out_ptr261,
                       long* out_ptr263,
                       float* out_ptr265,
                       float* out_ptr266,
                       long* out_ptr268,
                       float* out_ptr270,
                       float* out_ptr271,
                       long* out_ptr273,
                       float* out_ptr275,
                       float* out_ptr276,
                       long* out_ptr278,
                       float* out_ptr280,
                       float* out_ptr281,
                       long* out_ptr283,
                       float* out_ptr285,
                       float* out_ptr286,
                       long* out_ptr288,
                       float* out_ptr290,
                       float* out_ptr291,
                       long* out_ptr293,
                       float* out_ptr295,
                       float* out_ptr296,
                       long* out_ptr298,
                       float* out_ptr300,
                       float* out_ptr301,
                       long* out_ptr303,
                       float* out_ptr305,
                       float* out_ptr306,
                       long* out_ptr308)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = static_cast<float>(2.0);
            auto tmp4 = at::vec::Vectorized<float>(tmp3);
            auto tmp5 = tmp2 / tmp4;
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (3072L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (12L*x2) + (3072L*x0)), static_cast<long>(12L));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr1[static_cast<long>(x2 + (256L*x1) + (3072L*x0))];
                    out_ptr0[static_cast<long>(x1 + (12L*x2) + (3072L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (3072L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (12L*x2) + (3072L*x0)), static_cast<long>(12L));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr2[static_cast<long>(x2 + (256L*x1) + (3072L*x0))];
                    out_ptr1[static_cast<long>(x1 + (12L*x2) + (3072L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (3072L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (12L*x2) + (3072L*x0)), static_cast<long>(12L));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr3[static_cast<long>(x2 + (256L*x1) + (3072L*x0))];
                    out_ptr2[static_cast<long>(x1 + (12L*x2) + (3072L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (256L*x1) + (256L*x1_inner) + (3072L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (12L*x2) + (3072L*x0)), static_cast<long>(12L));
                }
            }
            #pragma GCC ivdep
            for(long x1=static_cast<long>(8L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr4[static_cast<long>(x2 + (256L*x1) + (3072L*x0))];
                    out_ptr3[static_cast<long>(x1 + (12L*x2) + (3072L*x0))] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (12544L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (16L*x2) + (12544L*x0)), static_cast<long>(16L));
                }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2400L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)), static_cast<long>(8L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(2400L); x2<static_cast<long>(2401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2400L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)), static_cast<long>(8L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(2400L); x2<static_cast<long>(2401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr7[static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2400L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr7 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)), static_cast<long>(8L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(2400L); x2<static_cast<long>(2401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr8[static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr7 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2400L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr8 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)), static_cast<long>(8L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(2400L); x2<static_cast<long>(2401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (2401L*x1) + (2401L*x1_inner) + (19208L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr8 + static_cast<long>(x1 + (8L*x2) + (19208L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9600L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (9604L*x1) + (9604L*x1_inner) + (76832L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr9 + static_cast<long>(x1 + (8L*x2) + (76832L*x0)), static_cast<long>(8L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(9600L); x2<static_cast<long>(9604L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (9604L*x1) + (9604L*x1_inner) + (76832L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr9 + static_cast<long>(x1 + (8L*x2) + (76832L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38416L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr11[static_cast<long>(x2 + (38416L*x1) + (153664L*x0))];
                        out_ptr10[static_cast<long>(x1 + (4L*x2) + (153664L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38416L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr12[static_cast<long>(x2 + (38416L*x1) + (153664L*x0))];
                        out_ptr11[static_cast<long>(x1 + (4L*x2) + (153664L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38416L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr13[static_cast<long>(x2 + (38416L*x1) + (153664L*x0))];
                        out_ptr12[static_cast<long>(x1 + (4L*x2) + (153664L*x0))] = tmp0;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38416L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr14[static_cast<long>(x2 + (38416L*x1) + (153664L*x0))];
                        out_ptr13[static_cast<long>(x1 + (4L*x2) + (153664L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr15 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr15 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr16 + static_cast<long>(x0));
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
                    tmp14.store(out_ptr16 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr18[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr18[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr19 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr20 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr21 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(25088.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0000398612827361);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr21 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr22[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr23[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr23 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr25 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr25 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr3 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr26 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(6272.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0001594642002871);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr26 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr26[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr28[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr27 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr30 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr30 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr4 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr31 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr31 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr30[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr33[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr31 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr35 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr35 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr5 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr36 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr36 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr34[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr38[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr35 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr40 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr40 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr6 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr41 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr41 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr38[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr43[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr39 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr45 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr45 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr7 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr46 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr46 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr42[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr48[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr43 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr50 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr50 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr8 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr51 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr51 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr46[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr53[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr47 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr55 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr55 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr9 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr56 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr56 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr50[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr58[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr51 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr60 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr60 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr10 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr61 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr61 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr54[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr63[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr55 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr65 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr65 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr11 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr66 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr66 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr58[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr68[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr59 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr70 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr70 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr12 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr71 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr71 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr62[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr73[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr63 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr75 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr75 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr13 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr76 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr76 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr66[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr78[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr67 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr80 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr80 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr14 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr81 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr81 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr70[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr83[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr71 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr85 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr85 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr15 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr86 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr86 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr74[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr88[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr75 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr90 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr90 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr16 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr91 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr91 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr78[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr93[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr79 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr95 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr95 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr17 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr96 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr96 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr82[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr98[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr83 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr100 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr100 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr18 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr101 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr101 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr86[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr103[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr87 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr105 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr105 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr19 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr106 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr106 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr90[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr108[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr91 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr110 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr110 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr20 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr111 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr111 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr94[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr113[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr95 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr115 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr115 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr21 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr116 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(1568.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0006381620931717);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr116 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr98[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr118[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr99 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr120 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr120 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr22 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr121 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr121 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr102[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr123[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr103 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr125 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr125 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr23 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr126 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr126 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr106[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr128[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr107 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr130 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr130 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr24 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr131 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr131 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr110[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr133[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr111 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr135 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr135 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr25 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr136 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr136 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr114[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr138[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr115 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr140 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr140 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr26 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr141 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr141 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr118[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr143[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr119 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr145 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr145 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr27 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr146 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr146 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr122[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr148[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr123 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr150 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr150 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr28 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr151 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr151 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr126[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr153[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr127 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr155 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr155 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr29 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr156 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr156 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr130[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr158[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr131 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr160 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr160 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr30 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr161 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr161 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr134[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr163[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr135 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr165 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr165 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr31 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr166 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr166 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr138[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr168[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr139 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr170 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr170 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr32 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr171 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr171 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr142[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr173[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr143 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr175 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr175 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr33 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr176 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr176 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr146[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr178[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr147 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr180 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr180 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr34 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr181 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr181 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr150[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr183[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr151 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr185 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr185 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr35 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr186 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr186 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr154[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr188[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr155 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr190 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr190 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr36 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr191 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr191 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr158[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr193[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr159 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr195 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr195 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr37 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr196 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr196 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr162[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr198[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr163 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr200 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr200 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr38 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr201 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr201 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr166[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr203[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr167 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr205 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr205 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr39 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr206 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr206 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr170[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr208[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr171 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr210 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr210 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr40 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr211 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr211 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr174[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr213[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr175 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr215 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr215 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr41 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr216 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr216 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr178[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr218[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr179 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr220 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr220 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr42 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr221 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(392.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0025575447570332);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr221 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr182[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr223[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr183 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr225 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr225 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr43 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr226 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr226 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr186[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr228[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr187 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr230 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr230 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr44 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr231 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr231 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr190[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr233[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr191 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr235 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr235 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr45 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr236 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr236 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr194[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr238[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr195 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr240 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr240 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr46 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr241 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr241 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr198[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr243[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr199 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr245 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr245 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr47 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr246 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr246 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr202[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr248[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr203 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr250 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr250 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr48 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr251 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr251 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr206[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr253[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr207 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr255 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr255 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr49 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr256 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr256 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr210[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr258[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr211 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr260 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr260 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr50 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr261 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr261 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr214[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr263[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr215 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr265 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr265 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr51 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr266 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr266 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr218[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr268[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr219 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr270 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr270 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr52 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr271 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr271 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr222[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr273[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr223 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr275 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr275 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr53 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr276 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr276 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr226[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr278[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr227 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr280 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr280 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr54 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr281 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr281 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr230[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr283[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr231 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr285 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr285 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr55 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr286 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr286 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr234[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr288[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr235 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr290 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr290 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr56 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr291 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr291 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr238[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr293[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr239 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr295 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr295 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr57 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr296 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr296 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr242[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr298[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr243 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr300 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr300 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr58 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr301 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr301 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr246[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr303[static_cast<long>(0L)] = tmp2;
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr247 + static_cast<long>(x0));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr305 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(0.1);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = static_cast<float>(0.9);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp8 = tmp3 + tmp7;
                    tmp8.store(out_ptr305 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr59 + static_cast<long>(x0));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr306 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(128.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    auto tmp4 = static_cast<float>(1.0078740157480315);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 * tmp5;
                    auto tmp7 = static_cast<float>(0.1);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 * tmp8;
                    auto tmp11 = static_cast<float>(0.9);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp14 = tmp9 + tmp13;
                    tmp14.store(out_ptr306 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                auto tmp0 = in_ptr250[static_cast<long>(0L)];
                auto tmp1 = static_cast<long>(1);
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr308[static_cast<long>(0L)] = tmp2;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_functional_add_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       float* in_out_ptr2,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const long* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const long* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const long* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13,
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const long* in_ptr18,
                       const long* in_ptr19,
                       float* out_ptr1,
                       float* out_ptr2,
                       long* out_ptr4,
                       float* out_ptr6,
                       float* out_ptr7,
                       long* out_ptr9,
                       float* out_ptr11,
                       float* out_ptr12,
                       long* out_ptr14,
                       float* out_ptr16,
                       float* out_ptr18,
                       float* out_ptr20,
                       float* out_ptr22,
                       long* out_ptr24,
                       long* out_ptr26)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr1 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0078740157480315);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr2 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr3[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr4[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr6 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr6 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr7 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0078740157480315);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr7 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr7[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr9[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr11 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            tmp8.store(out_ptr11 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr2 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr12 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(128.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.0078740157480315);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            tmp14.store(out_ptr12 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr11[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr14[static_cast<long>(0L)] = tmp2;
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr16 + static_cast<long>(x0));
            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr18 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.1);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 * tmp2;
            auto tmp5 = static_cast<float>(0.9);
            auto tmp6 = at::vec::Vectorized<float>(tmp5);
            auto tmp7 = tmp4 * tmp6;
            auto tmp8 = tmp3 + tmp7;
            auto tmp10 = tmp9 * tmp6;
            auto tmp11 = tmp3 + tmp10;
            tmp8.store(out_ptr16 + static_cast<long>(x0));
            tmp11.store(out_ptr18 + static_cast<long>(x0));
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr15 + static_cast<long>(x0));
            auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr20 + static_cast<long>(x0));
            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr22 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(8.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = tmp0 / tmp2;
            auto tmp4 = static_cast<float>(1.1428571428571428);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = tmp3 * tmp5;
            auto tmp7 = static_cast<float>(0.1);
            auto tmp8 = at::vec::Vectorized<float>(tmp7);
            auto tmp9 = tmp6 * tmp8;
            auto tmp11 = static_cast<float>(0.9);
            auto tmp12 = at::vec::Vectorized<float>(tmp11);
            auto tmp13 = tmp10 * tmp12;
            auto tmp14 = tmp9 + tmp13;
            auto tmp16 = tmp15 * tmp12;
            auto tmp17 = tmp9 + tmp16;
            tmp14.store(out_ptr20 + static_cast<long>(x0));
            tmp17.store(out_ptr22 + static_cast<long>(x0));
        }
    }
    {
        auto tmp0 = in_ptr18[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr24[static_cast<long>(0L)] = tmp2;
    }
    {
        auto tmp0 = in_ptr19[static_cast<long>(0L)];
        auto tmp1 = static_cast<long>(1);
        auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
        out_ptr26[static_cast<long>(0L)] = tmp2;
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415 = args
    args.clear()
    assert_size_stride(primals_1, (4, 196), (196, 1))
    assert_size_stride(primals_2, (4, 196), (196, 1))
    assert_size_stride(primals_3, (4, 196), (196, 1))
    assert_size_stride(primals_4, (4, 196), (196, 1))
    assert_size_stride(primals_5, (8, 196), (196, 1))
    assert_size_stride(primals_6, (8, 49), (49, 1))
    assert_size_stride(primals_7, (8, 49), (49, 1))
    assert_size_stride(primals_8, (8, 49), (49, 1))
    assert_size_stride(primals_9, (8, 49), (49, 1))
    assert_size_stride(primals_10, (16, 49), (49, 1))
    assert_size_stride(primals_11, (12, 16), (16, 1))
    assert_size_stride(primals_12, (12, 16), (16, 1))
    assert_size_stride(primals_13, (12, 16), (16, 1))
    assert_size_stride(primals_14, (12, 16), (16, 1))
    assert_size_stride(primals_15, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, ), (1, ))
    assert_size_stride(primals_21, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (256, 128), (128, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (128, 128), (128, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (256, 128), (128, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (128, 256), (256, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (256, 128), (128, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (128, 128), (128, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (256, 128), (128, 1))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (128, 256), (256, 1))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (256, 128), (128, 1))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (128, 128), (128, 1))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (256, 128), (128, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (128, 256), (256, 1))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (256, 128), (128, 1))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (128, 128), (128, 1))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (256, 128), (128, 1))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (128, 256), (256, 1))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (640, 128), (128, 1))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_77, (640, ), (1, ))
    assert_size_stride(primals_78, (128, 128), (128, 1))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (256, 512), (512, 1))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (512, 256), (256, 1))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (256, 512), (512, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (512, 256), (256, 1))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (256, 256), (256, 1))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (512, 256), (256, 1))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (256, 512), (512, 1))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (512, 256), (256, 1))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (256, 256), (256, 1))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (512, 256), (256, 1))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (256, 512), (512, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (512, 256), (256, 1))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (256, 256), (256, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (512, 256), (256, 1))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (256, 512), (512, 1))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (512, 256), (256, 1))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (256, 256), (256, 1))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (512, 256), (256, 1))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (256, 512), (512, 1))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (1280, 256), (256, 1))
    assert_size_stride(primals_139, (1280, ), (1, ))
    assert_size_stride(primals_140, (1280, ), (1, ))
    assert_size_stride(primals_141, (256, 256), (256, 1))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (384, 1024), (1024, 1))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (768, 384), (384, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (384, 768), (768, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (768, 384), (384, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (384, 384), (384, 1))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (768, 384), (384, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (384, 768), (768, 1))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (768, 384), (384, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (384, 384), (384, 1))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (768, 384), (384, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (384, 768), (768, 1))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, ), (1, ))
    assert_size_stride(primals_177, (768, 384), (384, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (384, 384), (384, 1))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (768, 384), (384, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (384, 768), (768, 1))
    assert_size_stride(primals_187, (384, ), (1, ))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (768, 384), (384, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (384, 384), (384, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (768, 384), (384, 1))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (384, 768), (768, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (1000, 384), (384, 1))
    assert_size_stride(primals_204, (1000, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (1000, 384), (384, 1))
    assert_size_stride(primals_208, (1000, ), (1, ))
    assert_size_stride(primals_209, (196, 196), (196, 1))
    assert_size_stride(primals_210, (196, 196), (196, 1))
    assert_size_stride(primals_211, (196, 196), (196, 1))
    assert_size_stride(primals_212, (196, 196), (196, 1))
    assert_size_stride(primals_213, (49, 196), (196, 1))
    assert_size_stride(primals_214, (49, 49), (49, 1))
    assert_size_stride(primals_215, (49, 49), (49, 1))
    assert_size_stride(primals_216, (49, 49), (49, 1))
    assert_size_stride(primals_217, (49, 49), (49, 1))
    assert_size_stride(primals_218, (16, 49), (49, 1))
    assert_size_stride(primals_219, (16, 16), (16, 1))
    assert_size_stride(primals_220, (16, 16), (16, 1))
    assert_size_stride(primals_221, (16, 16), (16, 1))
    assert_size_stride(primals_222, (16, 16), (16, 1))
    assert_size_stride(primals_223, (16, ), (1, ))
    assert_size_stride(primals_224, (16, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (256, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (256, ), (1, ))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (256, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (640, ), (1, ))
    assert_size_stride(primals_284, (640, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (256, ), (1, ))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (256, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (256, ), (1, ))
    assert_size_stride(primals_302, (256, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (256, ), (1, ))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (512, ), (1, ))
    assert_size_stride(primals_329, (512, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (256, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (512, ), (1, ))
    assert_size_stride(primals_335, (512, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (256, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (512, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (1280, ), (1, ))
    assert_size_stride(primals_347, (1280, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (256, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (384, ), (1, ))
    assert_size_stride(primals_353, (384, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (768, ), (1, ))
    assert_size_stride(primals_356, (768, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (384, ), (1, ))
    assert_size_stride(primals_359, (384, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (768, ), (1, ))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (384, ), (1, ))
    assert_size_stride(primals_365, (384, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (384, ), (1, ))
    assert_size_stride(primals_371, (384, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (768, ), (1, ))
    assert_size_stride(primals_374, (768, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (384, ), (1, ))
    assert_size_stride(primals_377, (384, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (768, ), (1, ))
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (384, ), (1, ))
    assert_size_stride(primals_383, (384, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (768, ), (1, ))
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (384, ), (1, ))
    assert_size_stride(primals_389, (384, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (768, ), (1, ))
    assert_size_stride(primals_392, (768, ), (1, ))
    assert_size_stride(primals_393, (), ())
    assert_size_stride(primals_394, (384, ), (1, ))
    assert_size_stride(primals_395, (384, ), (1, ))
    assert_size_stride(primals_396, (), ())
    assert_size_stride(primals_397, (768, ), (1, ))
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (), ())
    assert_size_stride(primals_400, (384, ), (1, ))
    assert_size_stride(primals_401, (384, ), (1, ))
    assert_size_stride(primals_402, (), ())
    assert_size_stride(primals_403, (768, ), (1, ))
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (), ())
    assert_size_stride(primals_406, (384, ), (1, ))
    assert_size_stride(primals_407, (384, ), (1, ))
    assert_size_stride(primals_408, (), ())
    assert_size_stride(primals_409, (384, ), (1, ))
    assert_size_stride(primals_410, (384, ), (1, ))
    assert_size_stride(primals_411, (), ())
    assert_size_stride(primals_412, (384, ), (1, ))
    assert_size_stride(primals_413, (384, ), (1, ))
    assert_size_stride(primals_414, (), ())
    assert_size_stride(primals_415, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_15.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(primals_415.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del primals_15
    del primals_18
    del primals_21
    del primals_24
    del primals_415
    # Source Nodes: [l__mod___stem_conv1_linear], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf4, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 16, 112, 112), (200704, 1, 1792, 16))
    buf6 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cpu', dtype=torch.float32)
    buf9 = empty((16, ), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    buf11 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_1(c_void_p(buf5.data_ptr()), c_void_p(primals_16.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_17
    # Source Nodes: [l__mod___stem_conv2_linear], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (8, 32, 56, 56), (100352, 1, 1792, 32))
    buf13 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    buf16 = empty((32, ), device='cpu', dtype=torch.float32)
    buf17 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    buf18 = empty_strided((8, 32, 56, 56), (100352, 1, 1792, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_2(c_void_p(buf12.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()))
    del primals_20
    # Source Nodes: [l__mod___stem_conv3_linear], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf18, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 64, 28, 28), (50176, 1, 1792, 64))
    buf20 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf21 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    buf23 = empty((64, ), device='cpu', dtype=torch.float32)
    buf24 = empty_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    buf25 = empty_strided((8, 64, 28, 28), (50176, 1, 1792, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_3(c_void_p(buf19.data_ptr()), c_void_p(primals_22.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()))
    del primals_23
    # Source Nodes: [l__mod___stem_conv4_linear], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 128, 14, 14), (25088, 1, 1792, 128))
    buf27 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf28 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cpu', dtype=torch.float32)
    buf30 = empty((128, ), device='cpu', dtype=torch.float32)
    buf31 = empty_strided((8, 128, 14, 14), (25088, 1, 1792, 128), device='cpu', dtype=torch.float32)
    buf32 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_clone_4(c_void_p(buf26.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(buf32.data_ptr()))
    del primals_26
    buf33 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_3], Original ATen: [aten.mm]
    extern_kernels.mm(buf32, reinterpret_tensor(primals_27, (128, 256), (1, 128), 0), out=buf33)
    buf34 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf35 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf37 = empty((256, ), device='cpu', dtype=torch.float32)
    buf38 = empty((8, 4, 196, 16), device='cpu', dtype=torch.float32)
    buf39 = empty((8, 4, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_5(c_void_p(buf33.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf39.data_ptr()))
    buf40 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf38, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf39, (32, 16, 196), (3136, 196, 1), 0), out=buf40)
    buf41 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf42 = reinterpret_tensor(buf40, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf40  # reuse
    buf43 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf44 = buf42; del buf42  # reuse
    buf45 = empty((8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_6(c_void_p(buf44.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf33.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(primals_28.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()))
    del primals_1
    del primals_29
    buf46 = empty((32, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf44, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf45, (32, 196, 32), (6272, 32, 1), 0), out=buf46)
    buf47 = empty((8, 196, 128), device='cpu', dtype=torch.float32)
    buf48 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_7(c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    buf49 = reinterpret_tensor(buf46, (1568, 128), (128, 1), 0); del buf46  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(buf48, reinterpret_tensor(primals_30, (128, 128), (1, 128), 0), out=buf49)
    buf50 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf51 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf53 = empty((128, ), device='cpu', dtype=torch.float32)
    buf54 = reinterpret_tensor(buf31, (8, 196, 128), (25088, 128, 1), 0); del buf31  # reuse
    buf55 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_8(c_void_p(buf54.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()))
    del primals_32
    buf56 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_8], Original ATen: [aten.mm]
    extern_kernels.mm(buf55, reinterpret_tensor(primals_33, (128, 256), (1, 128), 0), out=buf56)
    buf57 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf58 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf60 = empty((256, ), device='cpu', dtype=torch.float32)
    buf61 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    buf62 = empty((1568, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_9(c_void_p(buf56.data_ptr()), c_void_p(primals_34.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del primals_35
    buf63 = empty((1568, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_12], Original ATen: [aten.mm]
    extern_kernels.mm(buf62, reinterpret_tensor(primals_36, (256, 128), (1, 256), 0), out=buf63)
    buf64 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf65 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf67 = empty((128, ), device='cpu', dtype=torch.float32)
    buf68 = buf54; del buf54  # reuse
    buf69 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_10(c_void_p(buf68.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(primals_38.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_38
    buf70 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_15], Original ATen: [aten.mm]
    extern_kernels.mm(buf69, reinterpret_tensor(primals_39, (128, 256), (1, 128), 0), out=buf70)
    buf71 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf72 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf74 = empty((256, ), device='cpu', dtype=torch.float32)
    buf75 = empty((8, 4, 196, 16), device='cpu', dtype=torch.float32)
    buf76 = empty((8, 4, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_11(c_void_p(buf70.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()))
    buf77 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf75, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf76, (32, 16, 196), (3136, 196, 1), 0), out=buf77)
    buf78 = buf43; del buf43  # reuse
    buf79 = reinterpret_tensor(buf77, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf77  # reuse
    buf80 = buf41; del buf41  # reuse
    buf81 = buf79; del buf79  # reuse
    buf82 = empty((8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_12(c_void_p(buf81.data_ptr()), c_void_p(primals_210.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(primals_40.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()))
    del primals_2
    del primals_41
    buf83 = empty((32, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf82, (32, 196, 32), (6272, 32, 1), 0), out=buf83)
    buf84 = empty((8, 196, 128), device='cpu', dtype=torch.float32)
    buf85 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_13(c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    buf86 = reinterpret_tensor(buf83, (1568, 128), (128, 1), 0); del buf83  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.mm]
    extern_kernels.mm(buf85, reinterpret_tensor(primals_42, (128, 128), (1, 128), 0), out=buf86)
    buf87 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf88 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf90 = empty((128, ), device='cpu', dtype=torch.float32)
    buf91 = buf68; del buf68  # reuse
    buf92 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_14(c_void_p(buf91.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf92.data_ptr()))
    del primals_44
    buf93 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_20], Original ATen: [aten.mm]
    extern_kernels.mm(buf92, reinterpret_tensor(primals_45, (128, 256), (1, 128), 0), out=buf93)
    buf94 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf95 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf97 = empty((256, ), device='cpu', dtype=torch.float32)
    buf98 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    buf99 = empty((1568, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_15(c_void_p(buf93.data_ptr()), c_void_p(primals_46.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()))
    del primals_47
    buf100 = empty((1568, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_24], Original ATen: [aten.mm]
    extern_kernels.mm(buf99, reinterpret_tensor(primals_48, (256, 128), (1, 256), 0), out=buf100)
    buf101 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf102 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf104 = empty((128, ), device='cpu', dtype=torch.float32)
    buf105 = buf91; del buf91  # reuse
    buf106 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_16(c_void_p(buf105.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()))
    del primals_50
    buf107 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_27], Original ATen: [aten.mm]
    extern_kernels.mm(buf106, reinterpret_tensor(primals_51, (128, 256), (1, 128), 0), out=buf107)
    buf108 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf109 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf111 = empty((256, ), device='cpu', dtype=torch.float32)
    buf112 = empty((8, 4, 196, 16), device='cpu', dtype=torch.float32)
    buf113 = empty((8, 4, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_17(c_void_p(buf107.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    buf114 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf112, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf113, (32, 16, 196), (3136, 196, 1), 0), out=buf114)
    buf115 = buf80; del buf80  # reuse
    buf116 = reinterpret_tensor(buf114, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf114  # reuse
    buf117 = buf78; del buf78  # reuse
    buf118 = buf116; del buf116  # reuse
    buf119 = empty((8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_18(c_void_p(buf118.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(buf119.data_ptr()))
    del primals_3
    del primals_53
    buf120 = empty((32, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf118, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf119, (32, 196, 32), (6272, 32, 1), 0), out=buf120)
    buf121 = empty((8, 196, 128), device='cpu', dtype=torch.float32)
    buf122 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_19(c_void_p(buf120.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(buf122.data_ptr()))
    buf123 = reinterpret_tensor(buf120, (1568, 128), (128, 1), 0); del buf120  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(buf122, reinterpret_tensor(primals_54, (128, 128), (1, 128), 0), out=buf123)
    buf124 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf125 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf127 = empty((128, ), device='cpu', dtype=torch.float32)
    buf128 = buf105; del buf105  # reuse
    buf129 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_20(c_void_p(buf128.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_56.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()))
    del primals_56
    buf130 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(buf129, reinterpret_tensor(primals_57, (128, 256), (1, 128), 0), out=buf130)
    buf131 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf132 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf134 = empty((256, ), device='cpu', dtype=torch.float32)
    buf135 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    buf136 = empty((1568, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_21(c_void_p(buf130.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf136.data_ptr()))
    del primals_59
    buf137 = empty((1568, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_36], Original ATen: [aten.mm]
    extern_kernels.mm(buf136, reinterpret_tensor(primals_60, (256, 128), (1, 256), 0), out=buf137)
    buf138 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf139 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf141 = empty((128, ), device='cpu', dtype=torch.float32)
    buf142 = buf128; del buf128  # reuse
    buf143 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_22(c_void_p(buf142.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf143.data_ptr()))
    del primals_62
    buf144 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_39], Original ATen: [aten.mm]
    extern_kernels.mm(buf143, reinterpret_tensor(primals_63, (128, 256), (1, 128), 0), out=buf144)
    buf145 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf146 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf148 = empty((256, ), device='cpu', dtype=torch.float32)
    buf149 = empty((8, 4, 196, 16), device='cpu', dtype=torch.float32)
    buf150 = empty((8, 4, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_23(c_void_p(buf144.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(buf150.data_ptr()))
    buf151 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf150, (32, 16, 196), (3136, 196, 1), 0), out=buf151)
    buf152 = buf117; del buf117  # reuse
    buf153 = reinterpret_tensor(buf151, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf151  # reuse
    buf154 = buf115; del buf115  # reuse
    buf155 = buf153; del buf153  # reuse
    buf156 = empty((8, 4, 196, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_24(c_void_p(buf155.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(primals_64.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf156.data_ptr()))
    del buf152
    del buf154
    del primals_4
    del primals_65
    buf157 = empty((32, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf155, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf156, (32, 196, 32), (6272, 32, 1), 0), out=buf157)
    buf158 = empty((8, 196, 128), device='cpu', dtype=torch.float32)
    buf159 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_25(c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf159.data_ptr()))
    buf160 = reinterpret_tensor(buf157, (1568, 128), (128, 1), 0); del buf157  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.mm]
    extern_kernels.mm(buf159, reinterpret_tensor(primals_66, (128, 128), (1, 128), 0), out=buf160)
    buf161 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf162 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf164 = empty((128, ), device='cpu', dtype=torch.float32)
    buf165 = buf142; del buf142  # reuse
    buf166 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_26(c_void_p(buf165.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf166.data_ptr()))
    del primals_68
    buf167 = empty((1568, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_44], Original ATen: [aten.mm]
    extern_kernels.mm(buf166, reinterpret_tensor(primals_69, (128, 256), (1, 128), 0), out=buf167)
    buf168 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf169 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf171 = empty((256, ), device='cpu', dtype=torch.float32)
    buf172 = empty((8, 196, 256), device='cpu', dtype=torch.float32)
    buf173 = empty((1568, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_27(c_void_p(buf167.data_ptr()), c_void_p(primals_70.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()))
    del primals_71
    buf174 = empty((1568, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_48], Original ATen: [aten.mm]
    extern_kernels.mm(buf173, reinterpret_tensor(primals_72, (256, 128), (1, 256), 0), out=buf174)
    buf175 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf176 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf178 = empty((128, ), device='cpu', dtype=torch.float32)
    buf179 = buf165; del buf165  # reuse
    buf180 = empty((1568, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional__unsafe_view_add_clone_28(c_void_p(buf179.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(primals_74.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf180.data_ptr()))
    del primals_74
    buf181 = empty((1568, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_52], Original ATen: [aten.mm]
    extern_kernels.mm(buf180, reinterpret_tensor(primals_75, (128, 640), (1, 128), 0), out=buf181)
    buf182 = empty((1, 640), device='cpu', dtype=torch.float32)
    buf183 = empty((1, 640), device='cpu', dtype=torch.float32)
    buf185 = empty((640, ), device='cpu', dtype=torch.float32)
    buf186 = empty((392, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_view_29(c_void_p(buf181.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()))
    buf187 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_55], Original ATen: [aten.mm]
    extern_kernels.mm(buf186, reinterpret_tensor(primals_78, (128, 128), (1, 128), 0), out=buf187)
    buf188 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf189 = empty((1, 128), device='cpu', dtype=torch.float32)
    buf191 = empty((128, ), device='cpu', dtype=torch.float32)
    buf192 = empty((8, 8, 49, 16), device='cpu', dtype=torch.float32)
    buf193 = reinterpret_tensor(buf179, (8, 8, 16, 196), (25088, 3136, 196, 1), 0); del buf179  # reuse
    cpp_fused__native_batch_norm_legit_functional_clone_30(c_void_p(buf187.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del primals_80
    buf194 = empty((64, 49, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf192, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf193, (64, 16, 196), (3136, 196, 1), 0), out=buf194)
    buf195 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cpu', dtype=torch.float32)
    buf196 = reinterpret_tensor(buf194, (8, 8, 49, 196), (76832, 9604, 196, 1), 0); del buf194  # reuse
    buf197 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cpu', dtype=torch.float32)
    buf198 = buf196; del buf196  # reuse
    buf199 = empty((8, 8, 196, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_31(c_void_p(buf198.data_ptr()), c_void_p(primals_213.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(primals_76.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    del primals_5
    del primals_77
    buf200 = empty((64, 49, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf198, (64, 49, 196), (9604, 196, 1), 0), reinterpret_tensor(buf199, (64, 196, 64), (12544, 64, 1), 0), out=buf200)
    buf201 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf202 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_32(c_void_p(buf200.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    buf203 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_57], Original ATen: [aten.mm]
    extern_kernels.mm(buf202, reinterpret_tensor(primals_81, (512, 256), (1, 512), 0), out=buf203)
    buf204 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf205 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf207 = empty((256, ), device='cpu', dtype=torch.float32)
    buf208 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_33(c_void_p(buf203.data_ptr()), c_void_p(primals_82.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf208.data_ptr()))
    del primals_83
    buf209 = reinterpret_tensor(buf200, (392, 512), (512, 1), 0); del buf200  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf208, (392, 256), (256, 1), 0), reinterpret_tensor(primals_84, (256, 512), (1, 256), 0), out=buf209)
    buf210 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf211 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf213 = empty((512, ), device='cpu', dtype=torch.float32)
    buf214 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf215 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_34(c_void_p(buf209.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del primals_86
    buf216 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_64], Original ATen: [aten.mm]
    extern_kernels.mm(buf215, reinterpret_tensor(primals_87, (512, 256), (1, 512), 0), out=buf216)
    buf217 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf218 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf220 = empty((256, ), device='cpu', dtype=torch.float32)
    buf221 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_35(c_void_p(buf216.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    del primals_89
    buf222 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_68], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf221, (392, 256), (256, 1), 0), reinterpret_tensor(primals_90, (256, 512), (1, 256), 0), out=buf222)
    buf223 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf224 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf226 = empty((512, ), device='cpu', dtype=torch.float32)
    buf227 = empty((8, 8, 49, 16), device='cpu', dtype=torch.float32)
    buf228 = empty((8, 8, 16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_36(c_void_p(buf222.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()))
    buf229 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf228, (64, 16, 49), (784, 49, 1), 0), out=buf229)
    buf230 = buf197; del buf197  # reuse
    buf231 = reinterpret_tensor(buf229, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf229  # reuse
    buf232 = buf195; del buf195  # reuse
    buf233 = buf231; del buf231  # reuse
    buf234 = empty((8, 8, 49, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_37(c_void_p(buf233.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()))
    del primals_6
    del primals_92
    buf235 = empty((64, 49, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf233, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf234, (64, 49, 32), (1568, 32, 1), 0), out=buf235)
    buf236 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    buf237 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_38(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    buf238 = reinterpret_tensor(buf235, (392, 256), (256, 1), 0); del buf235  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.mm]
    extern_kernels.mm(buf237, reinterpret_tensor(primals_93, (256, 256), (1, 256), 0), out=buf238)
    buf239 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf240 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf242 = empty((256, ), device='cpu', dtype=torch.float32)
    buf243 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_39(c_void_p(buf238.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()))
    del primals_95
    buf244 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_73], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf243, (392, 256), (256, 1), 0), reinterpret_tensor(primals_96, (256, 512), (1, 256), 0), out=buf244)
    buf245 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf246 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf248 = empty((512, ), device='cpu', dtype=torch.float32)
    buf249 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf250 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_40(c_void_p(buf244.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()), c_void_p(buf250.data_ptr()))
    del primals_98
    buf251 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_77], Original ATen: [aten.mm]
    extern_kernels.mm(buf250, reinterpret_tensor(primals_99, (512, 256), (1, 512), 0), out=buf251)
    buf252 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf253 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf255 = empty((256, ), device='cpu', dtype=torch.float32)
    buf256 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_41(c_void_p(buf251.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf256.data_ptr()))
    del primals_101
    buf257 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (392, 256), (256, 1), 0), reinterpret_tensor(primals_102, (256, 512), (1, 256), 0), out=buf257)
    buf258 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf259 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf261 = empty((512, ), device='cpu', dtype=torch.float32)
    buf262 = empty((8, 8, 49, 16), device='cpu', dtype=torch.float32)
    buf263 = empty((8, 8, 16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_42(c_void_p(buf257.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    buf264 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf262, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf263, (64, 16, 49), (784, 49, 1), 0), out=buf264)
    buf265 = buf232; del buf232  # reuse
    buf266 = reinterpret_tensor(buf264, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf264  # reuse
    buf267 = buf230; del buf230  # reuse
    buf268 = buf266; del buf266  # reuse
    buf269 = empty((8, 8, 49, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_43(c_void_p(buf268.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()))
    del primals_104
    del primals_7
    buf270 = empty((64, 49, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf268, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf269, (64, 49, 32), (1568, 32, 1), 0), out=buf270)
    buf271 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    buf272 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_44(c_void_p(buf270.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(buf272.data_ptr()))
    buf273 = reinterpret_tensor(buf270, (392, 256), (256, 1), 0); del buf270  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.mm]
    extern_kernels.mm(buf272, reinterpret_tensor(primals_105, (256, 256), (1, 256), 0), out=buf273)
    buf274 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf275 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf277 = empty((256, ), device='cpu', dtype=torch.float32)
    buf278 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_45(c_void_p(buf273.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del primals_107
    buf279 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_85], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf278, (392, 256), (256, 1), 0), reinterpret_tensor(primals_108, (256, 512), (1, 256), 0), out=buf279)
    buf280 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf281 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf283 = empty((512, ), device='cpu', dtype=torch.float32)
    buf284 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf285 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_46(c_void_p(buf279.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_110.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()))
    del primals_110
    buf286 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_89], Original ATen: [aten.mm]
    extern_kernels.mm(buf285, reinterpret_tensor(primals_111, (512, 256), (1, 512), 0), out=buf286)
    buf287 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf288 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf290 = empty((256, ), device='cpu', dtype=torch.float32)
    buf291 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_47(c_void_p(buf286.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del primals_113
    buf292 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_92], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf291, (392, 256), (256, 1), 0), reinterpret_tensor(primals_114, (256, 512), (1, 256), 0), out=buf292)
    buf293 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf294 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf296 = empty((512, ), device='cpu', dtype=torch.float32)
    buf297 = empty((8, 8, 49, 16), device='cpu', dtype=torch.float32)
    buf298 = empty((8, 8, 16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_48(c_void_p(buf292.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf297.data_ptr()), c_void_p(buf298.data_ptr()))
    buf299 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf297, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf298, (64, 16, 49), (784, 49, 1), 0), out=buf299)
    buf300 = buf267; del buf267  # reuse
    buf301 = reinterpret_tensor(buf299, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf299  # reuse
    buf302 = buf265; del buf265  # reuse
    buf303 = buf301; del buf301  # reuse
    buf304 = empty((8, 8, 49, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_49(c_void_p(buf303.data_ptr()), c_void_p(primals_216.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_116.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf302.data_ptr()), c_void_p(buf304.data_ptr()))
    del primals_116
    del primals_8
    buf305 = empty((64, 49, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf303, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf304, (64, 49, 32), (1568, 32, 1), 0), out=buf305)
    buf306 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    buf307 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_50(c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(buf307.data_ptr()))
    buf308 = reinterpret_tensor(buf305, (392, 256), (256, 1), 0); del buf305  # reuse
    # Source Nodes: [x_94], Original ATen: [aten.mm]
    extern_kernels.mm(buf307, reinterpret_tensor(primals_117, (256, 256), (1, 256), 0), out=buf308)
    buf309 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf310 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf312 = empty((256, ), device='cpu', dtype=torch.float32)
    buf313 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_51(c_void_p(buf308.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf312.data_ptr()), c_void_p(buf313.data_ptr()))
    del primals_119
    buf314 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_97], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf313, (392, 256), (256, 1), 0), reinterpret_tensor(primals_120, (256, 512), (1, 256), 0), out=buf314)
    buf315 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf316 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf318 = empty((512, ), device='cpu', dtype=torch.float32)
    buf319 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf320 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_52(c_void_p(buf314.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_122.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()))
    del primals_122
    buf321 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_101], Original ATen: [aten.mm]
    extern_kernels.mm(buf320, reinterpret_tensor(primals_123, (512, 256), (1, 512), 0), out=buf321)
    buf322 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf323 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf325 = empty((256, ), device='cpu', dtype=torch.float32)
    buf326 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_53(c_void_p(buf321.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_125.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf326.data_ptr()))
    del primals_125
    buf327 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_104], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf326, (392, 256), (256, 1), 0), reinterpret_tensor(primals_126, (256, 512), (1, 256), 0), out=buf327)
    buf328 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf329 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf331 = empty((512, ), device='cpu', dtype=torch.float32)
    buf332 = empty((8, 8, 49, 16), device='cpu', dtype=torch.float32)
    buf333 = empty((8, 8, 16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_54(c_void_p(buf327.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    buf334 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf332, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf333, (64, 16, 49), (784, 49, 1), 0), out=buf334)
    buf335 = buf302; del buf302  # reuse
    buf336 = reinterpret_tensor(buf334, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf334  # reuse
    buf337 = buf300; del buf300  # reuse
    buf338 = buf336; del buf336  # reuse
    buf339 = empty((8, 8, 49, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_55(c_void_p(buf338.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_128.data_ptr()), c_void_p(buf335.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(buf339.data_ptr()))
    del buf335
    del buf337
    del primals_128
    del primals_9
    buf340 = empty((64, 49, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf338, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf339, (64, 49, 32), (1568, 32, 1), 0), out=buf340)
    buf341 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    buf342 = empty((392, 256), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_56(c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()), c_void_p(buf342.data_ptr()))
    buf343 = reinterpret_tensor(buf340, (392, 256), (256, 1), 0); del buf340  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.mm]
    extern_kernels.mm(buf342, reinterpret_tensor(primals_129, (256, 256), (1, 256), 0), out=buf343)
    buf344 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf345 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf347 = empty((256, ), device='cpu', dtype=torch.float32)
    buf348 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_57(c_void_p(buf343.data_ptr()), c_void_p(buf326.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()))
    del primals_131
    buf349 = empty((392, 512), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_109], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf348, (392, 256), (256, 1), 0), reinterpret_tensor(primals_132, (256, 512), (1, 256), 0), out=buf349)
    buf350 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf351 = empty((1, 512), device='cpu', dtype=torch.float32)
    buf353 = empty((512, ), device='cpu', dtype=torch.float32)
    buf354 = empty((8, 49, 512), device='cpu', dtype=torch.float32)
    buf355 = empty((392, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_58(c_void_p(buf349.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_134.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf355.data_ptr()))
    del primals_134
    buf356 = empty((392, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_113], Original ATen: [aten.mm]
    extern_kernels.mm(buf355, reinterpret_tensor(primals_135, (512, 256), (1, 512), 0), out=buf356)
    buf357 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf358 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf360 = empty((256, ), device='cpu', dtype=torch.float32)
    buf361 = empty((8, 49, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_59(c_void_p(buf356.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_137.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf358.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()))
    del primals_137
    buf362 = empty((392, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_117], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf361, (392, 256), (256, 1), 0), reinterpret_tensor(primals_138, (256, 1280), (1, 256), 0), out=buf362)
    buf363 = empty((1, 1280), device='cpu', dtype=torch.float32)
    buf364 = empty((1, 1280), device='cpu', dtype=torch.float32)
    buf366 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf367 = empty((128, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_view_60(c_void_p(buf362.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    buf368 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_120], Original ATen: [aten.mm]
    extern_kernels.mm(buf367, reinterpret_tensor(primals_141, (256, 256), (1, 256), 0), out=buf368)
    buf369 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf370 = empty((1, 256), device='cpu', dtype=torch.float32)
    buf372 = empty((256, ), device='cpu', dtype=torch.float32)
    buf373 = empty((8, 16, 16, 16), device='cpu', dtype=torch.float32)
    buf374 = empty((8, 16, 16, 49), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_61(c_void_p(buf368.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()))
    del primals_143
    buf375 = empty((128, 16, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf373, (128, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf374, (128, 16, 49), (784, 49, 1), 0), out=buf375)
    buf376 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cpu', dtype=torch.float32)
    buf377 = reinterpret_tensor(buf375, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf375  # reuse
    buf378 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cpu', dtype=torch.float32)
    buf379 = buf377; del buf377  # reuse
    buf380 = empty((8, 16, 49, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_62(c_void_p(buf379.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_10.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_140.data_ptr()), c_void_p(buf376.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(buf380.data_ptr()))
    del buf376
    del buf378
    del primals_10
    del primals_140
    buf381 = empty((128, 16, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf379, (128, 16, 49), (784, 49, 1), 0), reinterpret_tensor(buf380, (128, 49, 64), (3136, 64, 1), 0), out=buf381)
    buf382 = empty((8, 16, 1024), device='cpu', dtype=torch.float32)
    buf383 = empty((128, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_63(c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf383.data_ptr()))
    del buf381
    buf384 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_122], Original ATen: [aten.mm]
    extern_kernels.mm(buf383, reinterpret_tensor(primals_144, (1024, 384), (1, 1024), 0), out=buf384)
    buf385 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf386 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf388 = empty((384, ), device='cpu', dtype=torch.float32)
    buf389 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_64(c_void_p(buf384.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(buf386.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del primals_146
    buf390 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_125], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf389, (128, 384), (384, 1), 0), reinterpret_tensor(primals_147, (384, 768), (1, 384), 0), out=buf390)
    buf391 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf392 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf394 = empty((768, ), device='cpu', dtype=torch.float32)
    buf395 = empty((8, 16, 768), device='cpu', dtype=torch.float32)
    buf396 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_65(c_void_p(buf390.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_149.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()))
    del primals_149
    buf397 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_129], Original ATen: [aten.mm]
    extern_kernels.mm(buf396, reinterpret_tensor(primals_150, (768, 384), (1, 768), 0), out=buf397)
    buf398 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf399 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf401 = empty((384, ), device='cpu', dtype=torch.float32)
    buf402 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_66(c_void_p(buf397.data_ptr()), c_void_p(buf389.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_152.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(buf399.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(buf402.data_ptr()))
    del primals_152
    buf403 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_133], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf402, (128, 384), (384, 1), 0), reinterpret_tensor(primals_153, (384, 768), (1, 384), 0), out=buf403)
    buf404 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf405 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf407 = empty((768, ), device='cpu', dtype=torch.float32)
    buf408 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    buf409 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_67(c_void_p(buf403.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(buf407.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(buf409.data_ptr()))
    buf410 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf408, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf409, (96, 16, 16), (256, 16, 1), 0), out=buf410)
    buf411 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cpu', dtype=torch.float32)
    buf412 = reinterpret_tensor(buf410, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf410  # reuse
    buf413 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cpu', dtype=torch.float32)
    buf414 = buf412; del buf412  # reuse
    buf415 = empty((8, 12, 16, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_68(c_void_p(buf414.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()), c_void_p(primals_154.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf415.data_ptr()))
    del primals_11
    del primals_155
    buf416 = empty((96, 16, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf414, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf415, (96, 16, 32), (512, 32, 1), 0), out=buf416)
    buf417 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    buf418 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_69(c_void_p(buf416.data_ptr()), c_void_p(buf417.data_ptr()), c_void_p(buf418.data_ptr()))
    buf419 = reinterpret_tensor(buf416, (128, 384), (384, 1), 0); del buf416  # reuse
    # Source Nodes: [x_135], Original ATen: [aten.mm]
    extern_kernels.mm(buf418, reinterpret_tensor(primals_156, (384, 384), (1, 384), 0), out=buf419)
    buf420 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf421 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf423 = empty((384, ), device='cpu', dtype=torch.float32)
    buf424 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_70(c_void_p(buf419.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(primals_157.data_ptr()), c_void_p(primals_158.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()), c_void_p(buf423.data_ptr()), c_void_p(buf424.data_ptr()))
    del primals_158
    buf425 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_138], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf424, (128, 384), (384, 1), 0), reinterpret_tensor(primals_159, (384, 768), (1, 384), 0), out=buf425)
    buf426 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf427 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf429 = empty((768, ), device='cpu', dtype=torch.float32)
    buf430 = empty((8, 16, 768), device='cpu', dtype=torch.float32)
    buf431 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_71(c_void_p(buf425.data_ptr()), c_void_p(primals_160.data_ptr()), c_void_p(primals_161.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()), c_void_p(buf431.data_ptr()))
    del primals_161
    buf432 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_142], Original ATen: [aten.mm]
    extern_kernels.mm(buf431, reinterpret_tensor(primals_162, (768, 384), (1, 768), 0), out=buf432)
    buf433 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf434 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf436 = empty((384, ), device='cpu', dtype=torch.float32)
    buf437 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_72(c_void_p(buf432.data_ptr()), c_void_p(buf424.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(primals_164.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    del primals_164
    buf438 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_145], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf437, (128, 384), (384, 1), 0), reinterpret_tensor(primals_165, (384, 768), (1, 384), 0), out=buf438)
    buf439 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf440 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf442 = empty((768, ), device='cpu', dtype=torch.float32)
    buf443 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    buf444 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_73(c_void_p(buf438.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()))
    buf445 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf443, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf444, (96, 16, 16), (256, 16, 1), 0), out=buf445)
    buf446 = buf413; del buf413  # reuse
    buf447 = reinterpret_tensor(buf445, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf445  # reuse
    buf448 = buf411; del buf411  # reuse
    buf449 = buf447; del buf447  # reuse
    buf450 = empty((8, 12, 16, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_74(c_void_p(buf449.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf438.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(buf440.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf446.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(buf450.data_ptr()))
    del primals_12
    del primals_167
    buf451 = empty((96, 16, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf449, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf450, (96, 16, 32), (512, 32, 1), 0), out=buf451)
    buf452 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    buf453 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_75(c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    buf454 = reinterpret_tensor(buf451, (128, 384), (384, 1), 0); del buf451  # reuse
    # Source Nodes: [x_147], Original ATen: [aten.mm]
    extern_kernels.mm(buf453, reinterpret_tensor(primals_168, (384, 384), (1, 384), 0), out=buf454)
    buf455 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf456 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf458 = empty((384, ), device='cpu', dtype=torch.float32)
    buf459 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_76(c_void_p(buf454.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(primals_170.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(buf456.data_ptr()), c_void_p(buf458.data_ptr()), c_void_p(buf459.data_ptr()))
    del primals_170
    buf460 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_150], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf459, (128, 384), (384, 1), 0), reinterpret_tensor(primals_171, (384, 768), (1, 384), 0), out=buf460)
    buf461 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf462 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf464 = empty((768, ), device='cpu', dtype=torch.float32)
    buf465 = empty((8, 16, 768), device='cpu', dtype=torch.float32)
    buf466 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_77(c_void_p(buf460.data_ptr()), c_void_p(primals_172.data_ptr()), c_void_p(primals_173.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(buf465.data_ptr()), c_void_p(buf466.data_ptr()))
    del primals_173
    buf467 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_154], Original ATen: [aten.mm]
    extern_kernels.mm(buf466, reinterpret_tensor(primals_174, (768, 384), (1, 768), 0), out=buf467)
    buf468 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf469 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf471 = empty((384, ), device='cpu', dtype=torch.float32)
    buf472 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_78(c_void_p(buf467.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf471.data_ptr()), c_void_p(buf472.data_ptr()))
    del primals_176
    buf473 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_157], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf472, (128, 384), (384, 1), 0), reinterpret_tensor(primals_177, (384, 768), (1, 384), 0), out=buf473)
    buf474 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf475 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf477 = empty((768, ), device='cpu', dtype=torch.float32)
    buf478 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    buf479 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_79(c_void_p(buf473.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()), c_void_p(buf479.data_ptr()))
    buf480 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf478, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf479, (96, 16, 16), (256, 16, 1), 0), out=buf480)
    buf481 = buf448; del buf448  # reuse
    buf482 = reinterpret_tensor(buf480, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf480  # reuse
    buf483 = buf446; del buf446  # reuse
    buf484 = buf482; del buf482  # reuse
    buf485 = empty((8, 12, 16, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_80(c_void_p(buf484.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf473.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf481.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf485.data_ptr()))
    del primals_13
    del primals_179
    buf486 = empty((96, 16, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf484, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf485, (96, 16, 32), (512, 32, 1), 0), out=buf486)
    buf487 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    buf488 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_81(c_void_p(buf486.data_ptr()), c_void_p(buf487.data_ptr()), c_void_p(buf488.data_ptr()))
    buf489 = reinterpret_tensor(buf486, (128, 384), (384, 1), 0); del buf486  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.mm]
    extern_kernels.mm(buf488, reinterpret_tensor(primals_180, (384, 384), (1, 384), 0), out=buf489)
    buf490 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf491 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf493 = empty((384, ), device='cpu', dtype=torch.float32)
    buf494 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_82(c_void_p(buf489.data_ptr()), c_void_p(buf472.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    del primals_182
    buf495 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_162], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf494, (128, 384), (384, 1), 0), reinterpret_tensor(primals_183, (384, 768), (1, 384), 0), out=buf495)
    buf496 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf497 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf499 = empty((768, ), device='cpu', dtype=torch.float32)
    buf500 = empty((8, 16, 768), device='cpu', dtype=torch.float32)
    buf501 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_83(c_void_p(buf495.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    del primals_185
    buf502 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_166], Original ATen: [aten.mm]
    extern_kernels.mm(buf501, reinterpret_tensor(primals_186, (768, 384), (1, 768), 0), out=buf502)
    buf503 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf504 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf506 = empty((384, ), device='cpu', dtype=torch.float32)
    buf507 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_84(c_void_p(buf502.data_ptr()), c_void_p(buf494.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(buf504.data_ptr()), c_void_p(buf506.data_ptr()), c_void_p(buf507.data_ptr()))
    del primals_188
    buf508 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_169], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf507, (128, 384), (384, 1), 0), reinterpret_tensor(primals_189, (384, 768), (1, 384), 0), out=buf508)
    buf509 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf510 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf512 = empty((768, ), device='cpu', dtype=torch.float32)
    buf513 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    buf514 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_clone_85(c_void_p(buf508.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(buf513.data_ptr()), c_void_p(buf514.data_ptr()))
    buf515 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf513, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf514, (96, 16, 16), (256, 16, 1), 0), out=buf515)
    buf516 = buf483; del buf483  # reuse
    buf517 = reinterpret_tensor(buf515, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf515  # reuse
    buf518 = buf481; del buf481  # reuse
    buf519 = buf517; del buf517  # reuse
    buf520 = empty((8, 12, 16, 32), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_add_clone_index_mul_86(c_void_p(buf519.data_ptr()), c_void_p(primals_222.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf518.data_ptr()), c_void_p(buf520.data_ptr()))
    del buf516
    del buf518
    del primals_14
    del primals_191
    buf521 = empty((96, 16, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf519, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf520, (96, 16, 32), (512, 32, 1), 0), out=buf521)
    buf522 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    buf523 = empty((128, 384), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_view_clone_hardswish_view_87(c_void_p(buf521.data_ptr()), c_void_p(buf522.data_ptr()), c_void_p(buf523.data_ptr()))
    buf524 = reinterpret_tensor(buf521, (128, 384), (384, 1), 0); del buf521  # reuse
    # Source Nodes: [x_171], Original ATen: [aten.mm]
    extern_kernels.mm(buf523, reinterpret_tensor(primals_192, (384, 384), (1, 384), 0), out=buf524)
    buf525 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf526 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf528 = empty((384, ), device='cpu', dtype=torch.float32)
    buf529 = empty((8, 16, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_88(c_void_p(buf524.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(buf526.data_ptr()), c_void_p(buf528.data_ptr()), c_void_p(buf529.data_ptr()))
    del primals_194
    buf530 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_174], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf529, (128, 384), (384, 1), 0), reinterpret_tensor(primals_195, (384, 768), (1, 384), 0), out=buf530)
    buf531 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf532 = empty((1, 768), device='cpu', dtype=torch.float32)
    buf534 = empty((768, ), device='cpu', dtype=torch.float32)
    buf535 = empty((8, 16, 768), device='cpu', dtype=torch.float32)
    buf536 = empty((128, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_hardswish_view_89(c_void_p(buf530.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf536.data_ptr()))
    del primals_197
    buf537 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_178], Original ATen: [aten.mm]
    extern_kernels.mm(buf536, reinterpret_tensor(primals_198, (768, 384), (1, 768), 0), out=buf537)
    buf538 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf539 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf541 = empty((384, ), device='cpu', dtype=torch.float32)
    buf542 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf543 = buf542; del buf542  # reuse
    buf544 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf545 = empty((1, 384), device='cpu', dtype=torch.float32)
    buf547 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf549 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_functional_add_mean_90(c_void_p(buf543.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf529.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_201.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf541.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(buf547.data_ptr()), c_void_p(buf549.data_ptr()))
    del primals_200
    del primals_202
    del primals_206
    buf548 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_185], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_204, buf547, reinterpret_tensor(primals_203, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf548)
    del primals_204
    buf550 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_dist], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_208, buf549, reinterpret_tensor(primals_207, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf550)
    del primals_208
    buf551 = buf548; del buf548  # reuse
    buf552 = empty_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    buf553 = empty_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    buf554 = empty_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    buf555 = empty_strided((8, 12, 16, 16), (3072, 1, 192, 12), device='cpu', dtype=torch.float32)
    buf556 = empty_strided((8, 16, 16, 49), (12544, 1, 784, 16), device='cpu', dtype=torch.float32)
    buf557 = empty_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    buf558 = empty_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    buf559 = empty_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    buf560 = empty_strided((8, 8, 49, 49), (19208, 1, 392, 8), device='cpu', dtype=torch.float32)
    buf561 = empty_strided((8, 8, 49, 196), (76832, 1, 1568, 8), device='cpu', dtype=torch.float32)
    buf562 = empty_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    buf563 = empty_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    buf564 = empty_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    buf565 = empty_strided((8, 4, 196, 196), (153664, 1, 784, 4), device='cpu', dtype=torch.float32)
    buf570 = reinterpret_tensor(buf7, (16, ), (1, ), 0); del buf7  # reuse
    buf578 = reinterpret_tensor(buf14, (32, ), (1, ), 0); del buf14  # reuse
    buf586 = reinterpret_tensor(buf21, (64, ), (1, ), 0); del buf21  # reuse
    buf594 = reinterpret_tensor(buf28, (128, ), (1, ), 0); del buf28  # reuse
    buf602 = reinterpret_tensor(buf35, (256, ), (1, ), 0); del buf35  # reuse
    buf610 = reinterpret_tensor(buf51, (128, ), (1, ), 0); del buf51  # reuse
    buf618 = reinterpret_tensor(buf58, (256, ), (1, ), 0); del buf58  # reuse
    buf626 = reinterpret_tensor(buf65, (128, ), (1, ), 0); del buf65  # reuse
    buf634 = reinterpret_tensor(buf72, (256, ), (1, ), 0); del buf72  # reuse
    buf642 = reinterpret_tensor(buf88, (128, ), (1, ), 0); del buf88  # reuse
    buf650 = reinterpret_tensor(buf95, (256, ), (1, ), 0); del buf95  # reuse
    buf658 = reinterpret_tensor(buf102, (128, ), (1, ), 0); del buf102  # reuse
    buf666 = reinterpret_tensor(buf109, (256, ), (1, ), 0); del buf109  # reuse
    buf674 = reinterpret_tensor(buf125, (128, ), (1, ), 0); del buf125  # reuse
    buf682 = reinterpret_tensor(buf132, (256, ), (1, ), 0); del buf132  # reuse
    buf690 = reinterpret_tensor(buf139, (128, ), (1, ), 0); del buf139  # reuse
    buf698 = reinterpret_tensor(buf146, (256, ), (1, ), 0); del buf146  # reuse
    buf706 = reinterpret_tensor(buf162, (128, ), (1, ), 0); del buf162  # reuse
    buf714 = reinterpret_tensor(buf169, (256, ), (1, ), 0); del buf169  # reuse
    buf722 = reinterpret_tensor(buf176, (128, ), (1, ), 0); del buf176  # reuse
    buf730 = reinterpret_tensor(buf183, (640, ), (1, ), 0); del buf183  # reuse
    buf738 = reinterpret_tensor(buf189, (128, ), (1, ), 0); del buf189  # reuse
    buf746 = reinterpret_tensor(buf205, (256, ), (1, ), 0); del buf205  # reuse
    buf754 = reinterpret_tensor(buf211, (512, ), (1, ), 0); del buf211  # reuse
    buf762 = reinterpret_tensor(buf218, (256, ), (1, ), 0); del buf218  # reuse
    buf770 = reinterpret_tensor(buf224, (512, ), (1, ), 0); del buf224  # reuse
    buf778 = reinterpret_tensor(buf240, (256, ), (1, ), 0); del buf240  # reuse
    buf786 = reinterpret_tensor(buf246, (512, ), (1, ), 0); del buf246  # reuse
    buf794 = reinterpret_tensor(buf253, (256, ), (1, ), 0); del buf253  # reuse
    buf802 = reinterpret_tensor(buf259, (512, ), (1, ), 0); del buf259  # reuse
    buf810 = reinterpret_tensor(buf275, (256, ), (1, ), 0); del buf275  # reuse
    buf818 = reinterpret_tensor(buf281, (512, ), (1, ), 0); del buf281  # reuse
    buf826 = reinterpret_tensor(buf288, (256, ), (1, ), 0); del buf288  # reuse
    buf834 = reinterpret_tensor(buf294, (512, ), (1, ), 0); del buf294  # reuse
    buf842 = reinterpret_tensor(buf310, (256, ), (1, ), 0); del buf310  # reuse
    buf850 = reinterpret_tensor(buf316, (512, ), (1, ), 0); del buf316  # reuse
    buf858 = reinterpret_tensor(buf323, (256, ), (1, ), 0); del buf323  # reuse
    buf866 = reinterpret_tensor(buf329, (512, ), (1, ), 0); del buf329  # reuse
    buf874 = reinterpret_tensor(buf345, (256, ), (1, ), 0); del buf345  # reuse
    buf882 = reinterpret_tensor(buf351, (512, ), (1, ), 0); del buf351  # reuse
    buf890 = reinterpret_tensor(buf358, (256, ), (1, ), 0); del buf358  # reuse
    buf898 = reinterpret_tensor(buf364, (1280, ), (1, ), 0); del buf364  # reuse
    buf906 = reinterpret_tensor(buf370, (256, ), (1, ), 0); del buf370  # reuse
    buf914 = reinterpret_tensor(buf386, (384, ), (1, ), 0); del buf386  # reuse
    buf922 = reinterpret_tensor(buf392, (768, ), (1, ), 0); del buf392  # reuse
    buf930 = reinterpret_tensor(buf399, (384, ), (1, ), 0); del buf399  # reuse
    buf938 = reinterpret_tensor(buf405, (768, ), (1, ), 0); del buf405  # reuse
    buf946 = reinterpret_tensor(buf421, (384, ), (1, ), 0); del buf421  # reuse
    buf954 = reinterpret_tensor(buf427, (768, ), (1, ), 0); del buf427  # reuse
    buf962 = reinterpret_tensor(buf434, (384, ), (1, ), 0); del buf434  # reuse
    buf970 = reinterpret_tensor(buf440, (768, ), (1, ), 0); del buf440  # reuse
    buf978 = reinterpret_tensor(buf456, (384, ), (1, ), 0); del buf456  # reuse
    buf986 = reinterpret_tensor(buf462, (768, ), (1, ), 0); del buf462  # reuse
    buf994 = reinterpret_tensor(buf469, (384, ), (1, ), 0); del buf469  # reuse
    buf1002 = reinterpret_tensor(buf475, (768, ), (1, ), 0); del buf475  # reuse
    buf1010 = reinterpret_tensor(buf491, (384, ), (1, ), 0); del buf491  # reuse
    buf1018 = reinterpret_tensor(buf497, (768, ), (1, ), 0); del buf497  # reuse
    buf1026 = reinterpret_tensor(buf504, (384, ), (1, ), 0); del buf504  # reuse
    buf1034 = reinterpret_tensor(buf510, (768, ), (1, ), 0); del buf510  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_detach_div_91(c_void_p(buf551.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf578.data_ptr()), c_void_p(buf586.data_ptr()), c_void_p(buf594.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(buf610.data_ptr()), c_void_p(buf618.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(buf634.data_ptr()), c_void_p(buf642.data_ptr()), c_void_p(buf650.data_ptr()), c_void_p(buf658.data_ptr()), c_void_p(buf666.data_ptr()), c_void_p(buf674.data_ptr()), c_void_p(buf682.data_ptr()), c_void_p(buf690.data_ptr()), c_void_p(buf698.data_ptr()), c_void_p(buf706.data_ptr()), c_void_p(buf714.data_ptr()), c_void_p(buf722.data_ptr()), c_void_p(buf730.data_ptr()), c_void_p(buf738.data_ptr()), c_void_p(buf746.data_ptr()), c_void_p(buf754.data_ptr()), c_void_p(buf762.data_ptr()), c_void_p(buf770.data_ptr()), c_void_p(buf778.data_ptr()), c_void_p(buf786.data_ptr()), c_void_p(buf794.data_ptr()), c_void_p(buf802.data_ptr()), c_void_p(buf810.data_ptr()), c_void_p(buf818.data_ptr()), c_void_p(buf826.data_ptr()), c_void_p(buf834.data_ptr()), c_void_p(buf842.data_ptr()), c_void_p(buf850.data_ptr()), c_void_p(buf858.data_ptr()), c_void_p(buf866.data_ptr()), c_void_p(buf874.data_ptr()), c_void_p(buf882.data_ptr()), c_void_p(buf890.data_ptr()), c_void_p(buf898.data_ptr()), c_void_p(buf906.data_ptr()), c_void_p(buf914.data_ptr()), c_void_p(buf922.data_ptr()), c_void_p(buf930.data_ptr()), c_void_p(buf938.data_ptr()), c_void_p(buf946.data_ptr()), c_void_p(buf954.data_ptr()), c_void_p(buf962.data_ptr()), c_void_p(buf970.data_ptr()), c_void_p(buf978.data_ptr()), c_void_p(buf986.data_ptr()), c_void_p(buf994.data_ptr()), c_void_p(buf1002.data_ptr()), c_void_p(buf1010.data_ptr()), c_void_p(buf1018.data_ptr()), c_void_p(buf1026.data_ptr()), c_void_p(buf1034.data_ptr()), c_void_p(buf550.data_ptr()), c_void_p(buf519.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf449.data_ptr()), c_void_p(buf414.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf344.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(buf391.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(buf426.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(buf433.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(buf439.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(buf455.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(buf503.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()), c_void_p(buf552.data_ptr()), c_void_p(buf553.data_ptr()), c_void_p(buf554.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(buf556.data_ptr()), c_void_p(buf557.data_ptr()), c_void_p(buf558.data_ptr()), c_void_p(buf559.data_ptr()), c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(buf562.data_ptr()), c_void_p(buf563.data_ptr()), c_void_p(buf564.data_ptr()), c_void_p(buf565.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_225.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_228.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_234.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(primals_237.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_240.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_246.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_249.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_252.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_258.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_261.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_264.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_389.data_ptr()), c_void_p(primals_390.data_ptr()), c_void_p(primals_391.data_ptr()), c_void_p(primals_392.data_ptr()), c_void_p(primals_393.data_ptr()), c_void_p(primals_394.data_ptr()), c_void_p(primals_395.data_ptr()), c_void_p(primals_396.data_ptr()), c_void_p(primals_397.data_ptr()), c_void_p(primals_398.data_ptr()), c_void_p(primals_399.data_ptr()))
    del buf1002
    del buf1010
    del buf1018
    del buf1026
    del buf1034
    del buf550
    del buf570
    del buf578
    del buf586
    del buf594
    del buf602
    del buf610
    del buf618
    del buf626
    del buf634
    del buf642
    del buf650
    del buf658
    del buf666
    del buf674
    del buf682
    del buf690
    del buf698
    del buf706
    del buf714
    del buf722
    del buf730
    del buf738
    del buf746
    del buf754
    del buf762
    del buf770
    del buf778
    del buf786
    del buf794
    del buf802
    del buf810
    del buf818
    del buf826
    del buf834
    del buf842
    del buf850
    del buf858
    del buf866
    del buf874
    del buf882
    del buf890
    del buf898
    del buf906
    del buf914
    del buf922
    del buf930
    del buf938
    del buf946
    del buf954
    del buf962
    del buf970
    del buf978
    del buf986
    del buf994
    del primals_223
    del primals_224
    del primals_225
    del primals_226
    del primals_227
    del primals_228
    del primals_229
    del primals_230
    del primals_231
    del primals_232
    del primals_233
    del primals_234
    del primals_235
    del primals_236
    del primals_237
    del primals_238
    del primals_239
    del primals_240
    del primals_241
    del primals_242
    del primals_243
    del primals_244
    del primals_245
    del primals_246
    del primals_247
    del primals_248
    del primals_249
    del primals_250
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
    del primals_261
    del primals_262
    del primals_263
    del primals_264
    del primals_265
    del primals_266
    del primals_267
    del primals_268
    del primals_269
    del primals_270
    del primals_271
    del primals_272
    del primals_273
    del primals_274
    del primals_275
    del primals_276
    del primals_277
    del primals_278
    del primals_279
    del primals_280
    del primals_281
    del primals_282
    del primals_283
    del primals_284
    del primals_285
    del primals_286
    del primals_287
    del primals_288
    del primals_289
    del primals_290
    del primals_291
    del primals_292
    del primals_293
    del primals_294
    del primals_295
    del primals_296
    del primals_297
    del primals_298
    del primals_299
    del primals_300
    del primals_301
    del primals_302
    del primals_303
    del primals_304
    del primals_305
    del primals_306
    del primals_307
    del primals_308
    del primals_309
    del primals_310
    del primals_311
    del primals_312
    del primals_313
    del primals_314
    del primals_315
    del primals_316
    del primals_317
    del primals_318
    del primals_319
    del primals_320
    del primals_321
    del primals_322
    del primals_323
    del primals_324
    del primals_325
    del primals_326
    del primals_327
    del primals_328
    del primals_329
    del primals_330
    del primals_331
    del primals_332
    del primals_333
    del primals_334
    del primals_335
    del primals_336
    del primals_337
    del primals_338
    del primals_339
    del primals_340
    del primals_341
    del primals_342
    del primals_343
    del primals_344
    del primals_345
    del primals_346
    del primals_347
    del primals_348
    del primals_349
    del primals_350
    del primals_351
    del primals_352
    del primals_353
    del primals_354
    del primals_355
    del primals_356
    del primals_357
    del primals_358
    del primals_359
    del primals_360
    del primals_361
    del primals_362
    del primals_363
    del primals_364
    del primals_365
    del primals_366
    del primals_367
    del primals_368
    del primals_369
    del primals_370
    del primals_371
    del primals_372
    del primals_373
    del primals_374
    del primals_375
    del primals_376
    del primals_377
    del primals_378
    del primals_379
    del primals_380
    del primals_381
    del primals_382
    del primals_383
    del primals_384
    del primals_385
    del primals_386
    del primals_387
    del primals_388
    del primals_389
    del primals_390
    del primals_391
    del primals_392
    del primals_393
    del primals_394
    del primals_395
    del primals_396
    del primals_397
    del primals_398
    del primals_399
    buf1042 = reinterpret_tensor(buf526, (384, ), (1, ), 0); del buf526  # reuse
    buf1050 = reinterpret_tensor(buf532, (768, ), (1, ), 0); del buf532  # reuse
    buf1058 = reinterpret_tensor(buf539, (384, ), (1, ), 0); del buf539  # reuse
    cpp_fused__native_batch_norm_legit_functional_add_92(c_void_p(buf1042.data_ptr()), c_void_p(buf1050.data_ptr()), c_void_p(buf1058.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(buf531.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(buf538.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(buf544.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_414.data_ptr()), c_void_p(primals_400.data_ptr()), c_void_p(primals_401.data_ptr()), c_void_p(primals_402.data_ptr()), c_void_p(primals_403.data_ptr()), c_void_p(primals_404.data_ptr()), c_void_p(primals_405.data_ptr()), c_void_p(primals_406.data_ptr()), c_void_p(primals_407.data_ptr()), c_void_p(primals_408.data_ptr()), c_void_p(primals_409.data_ptr()), c_void_p(primals_412.data_ptr()), c_void_p(primals_410.data_ptr()), c_void_p(primals_413.data_ptr()), c_void_p(primals_411.data_ptr()), c_void_p(primals_414.data_ptr()))
    del buf1042
    del buf1050
    del buf1058
    del buf544
    del buf545
    del primals_400
    del primals_401
    del primals_402
    del primals_403
    del primals_404
    del primals_405
    del primals_406
    del primals_407
    del primals_408
    del primals_409
    del primals_410
    del primals_411
    del primals_412
    del primals_413
    del primals_414
    return (buf551, buf0, primals_16, buf1, primals_19, buf2, primals_22, buf3, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, buf4, buf5, buf9, buf10, buf11, buf12, buf16, buf17, buf18, buf19, buf23, buf24, buf25, buf26, buf30, buf32, buf33, buf37, buf47, buf48, buf49, buf53, buf55, buf56, buf60, buf61, buf62, buf63, buf67, buf69, buf70, buf74, buf84, buf85, buf86, buf90, buf92, buf93, buf97, buf98, buf99, buf100, buf104, buf106, buf107, buf111, buf121, buf122, buf123, buf127, buf129, buf130, buf134, buf135, buf136, buf137, buf141, buf143, buf144, buf148, buf158, buf159, buf160, buf164, buf166, buf167, buf171, buf172, buf173, buf174, buf178, buf180, buf181, buf185, buf186, buf187, buf191, buf201, buf202, buf203, buf207, reinterpret_tensor(buf208, (392, 256), (256, 1), 0), buf209, buf213, buf214, buf215, buf216, buf220, reinterpret_tensor(buf221, (392, 256), (256, 1), 0), buf222, buf226, buf236, buf237, buf238, buf242, reinterpret_tensor(buf243, (392, 256), (256, 1), 0), buf244, buf248, buf249, buf250, buf251, buf255, reinterpret_tensor(buf256, (392, 256), (256, 1), 0), buf257, buf261, buf271, buf272, buf273, buf277, reinterpret_tensor(buf278, (392, 256), (256, 1), 0), buf279, buf283, buf284, buf285, buf286, buf290, reinterpret_tensor(buf291, (392, 256), (256, 1), 0), buf292, buf296, buf306, buf307, buf308, buf312, reinterpret_tensor(buf313, (392, 256), (256, 1), 0), buf314, buf318, buf319, buf320, buf321, buf325, reinterpret_tensor(buf326, (392, 256), (256, 1), 0), buf327, buf331, buf341, buf342, buf343, buf347, reinterpret_tensor(buf348, (392, 256), (256, 1), 0), buf349, buf353, buf354, buf355, buf356, buf360, reinterpret_tensor(buf361, (392, 256), (256, 1), 0), buf362, buf366, buf367, buf368, buf372, buf382, buf383, buf384, buf388, reinterpret_tensor(buf389, (128, 384), (384, 1), 0), buf390, buf394, buf395, buf396, buf397, buf401, reinterpret_tensor(buf402, (128, 384), (384, 1), 0), buf403, buf407, buf417, buf418, buf419, buf423, reinterpret_tensor(buf424, (128, 384), (384, 1), 0), buf425, buf429, buf430, buf431, buf432, buf436, reinterpret_tensor(buf437, (128, 384), (384, 1), 0), buf438, buf442, buf452, buf453, buf454, buf458, reinterpret_tensor(buf459, (128, 384), (384, 1), 0), buf460, buf464, buf465, buf466, buf467, buf471, reinterpret_tensor(buf472, (128, 384), (384, 1), 0), buf473, buf477, buf487, buf488, buf489, buf493, reinterpret_tensor(buf494, (128, 384), (384, 1), 0), buf495, buf499, buf500, buf501, buf502, buf506, reinterpret_tensor(buf507, (128, 384), (384, 1), 0), buf508, buf512, buf522, buf523, buf524, buf528, reinterpret_tensor(buf529, (128, 384), (384, 1), 0), buf530, buf534, buf535, buf536, buf537, buf541, buf543, buf547, buf549, reinterpret_tensor(primals_207, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_203, (1000, 384), (384, 1), 0), reinterpret_tensor(buf538, (1, 384), (384, 1), 0), reinterpret_tensor(primals_198, (384, 768), (768, 1), 0), reinterpret_tensor(buf531, (1, 768), (768, 1), 0), reinterpret_tensor(primals_195, (768, 384), (384, 1), 0), reinterpret_tensor(buf525, (1, 384), (384, 1), 0), reinterpret_tensor(primals_192, (384, 384), (384, 1), 0), reinterpret_tensor(buf519, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf520, (96, 32, 16), (512, 1, 32), 0), buf552, reinterpret_tensor(buf513, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf514, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf509, (1, 768), (768, 1), 0), reinterpret_tensor(primals_189, (768, 384), (384, 1), 0), reinterpret_tensor(buf503, (1, 384), (384, 1), 0), reinterpret_tensor(primals_186, (384, 768), (768, 1), 0), reinterpret_tensor(buf496, (1, 768), (768, 1), 0), reinterpret_tensor(primals_183, (768, 384), (384, 1), 0), reinterpret_tensor(buf490, (1, 384), (384, 1), 0), reinterpret_tensor(primals_180, (384, 384), (384, 1), 0), reinterpret_tensor(buf484, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf485, (96, 32, 16), (512, 1, 32), 0), buf553, reinterpret_tensor(buf478, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf479, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf474, (1, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 384), (384, 1), 0), reinterpret_tensor(buf468, (1, 384), (384, 1), 0), reinterpret_tensor(primals_174, (384, 768), (768, 1), 0), reinterpret_tensor(buf461, (1, 768), (768, 1), 0), reinterpret_tensor(primals_171, (768, 384), (384, 1), 0), reinterpret_tensor(buf455, (1, 384), (384, 1), 0), reinterpret_tensor(primals_168, (384, 384), (384, 1), 0), reinterpret_tensor(buf449, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf450, (96, 32, 16), (512, 1, 32), 0), buf554, reinterpret_tensor(buf443, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf444, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf439, (1, 768), (768, 1), 0), reinterpret_tensor(primals_165, (768, 384), (384, 1), 0), reinterpret_tensor(buf433, (1, 384), (384, 1), 0), reinterpret_tensor(primals_162, (384, 768), (768, 1), 0), reinterpret_tensor(buf426, (1, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 384), (384, 1), 0), reinterpret_tensor(buf420, (1, 384), (384, 1), 0), reinterpret_tensor(primals_156, (384, 384), (384, 1), 0), reinterpret_tensor(buf414, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf415, (96, 32, 16), (512, 1, 32), 0), buf555, reinterpret_tensor(buf408, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf409, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf404, (1, 768), (768, 1), 0), reinterpret_tensor(primals_153, (768, 384), (384, 1), 0), reinterpret_tensor(buf398, (1, 384), (384, 1), 0), reinterpret_tensor(primals_150, (384, 768), (768, 1), 0), reinterpret_tensor(buf391, (1, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 384), (384, 1), 0), reinterpret_tensor(buf385, (1, 384), (384, 1), 0), reinterpret_tensor(primals_144, (384, 1024), (1024, 1), 0), reinterpret_tensor(buf379, (128, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf380, (128, 64, 49), (3136, 1, 64), 0), buf556, reinterpret_tensor(buf373, (128, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf374, (128, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf369, (1, 256), (256, 1), 0), reinterpret_tensor(primals_141, (256, 256), (256, 1), 0), reinterpret_tensor(buf363, (1, 1280), (1280, 1), 0), reinterpret_tensor(primals_138, (1280, 256), (256, 1), 0), reinterpret_tensor(buf357, (1, 256), (256, 1), 0), reinterpret_tensor(primals_135, (256, 512), (512, 1), 0), reinterpret_tensor(buf350, (1, 512), (512, 1), 0), reinterpret_tensor(primals_132, (512, 256), (256, 1), 0), reinterpret_tensor(buf344, (1, 256), (256, 1), 0), reinterpret_tensor(primals_129, (256, 256), (256, 1), 0), reinterpret_tensor(buf338, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf339, (64, 32, 49), (1568, 1, 32), 0), buf557, reinterpret_tensor(buf332, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf333, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf328, (1, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 256), (256, 1), 0), reinterpret_tensor(buf322, (1, 256), (256, 1), 0), reinterpret_tensor(primals_123, (256, 512), (512, 1), 0), reinterpret_tensor(buf315, (1, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 256), (256, 1), 0), reinterpret_tensor(buf309, (1, 256), (256, 1), 0), reinterpret_tensor(primals_117, (256, 256), (256, 1), 0), reinterpret_tensor(buf303, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf304, (64, 32, 49), (1568, 1, 32), 0), buf558, reinterpret_tensor(buf297, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf298, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf293, (1, 512), (512, 1), 0), reinterpret_tensor(primals_114, (512, 256), (256, 1), 0), reinterpret_tensor(buf287, (1, 256), (256, 1), 0), reinterpret_tensor(primals_111, (256, 512), (512, 1), 0), reinterpret_tensor(buf280, (1, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 256), (256, 1), 0), reinterpret_tensor(buf274, (1, 256), (256, 1), 0), reinterpret_tensor(primals_105, (256, 256), (256, 1), 0), reinterpret_tensor(buf268, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf269, (64, 32, 49), (1568, 1, 32), 0), buf559, reinterpret_tensor(buf262, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf263, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf258, (1, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 256), (256, 1), 0), reinterpret_tensor(buf252, (1, 256), (256, 1), 0), reinterpret_tensor(primals_99, (256, 512), (512, 1), 0), reinterpret_tensor(buf245, (1, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 256), (256, 1), 0), reinterpret_tensor(buf239, (1, 256), (256, 1), 0), reinterpret_tensor(primals_93, (256, 256), (256, 1), 0), reinterpret_tensor(buf233, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf234, (64, 32, 49), (1568, 1, 32), 0), buf560, reinterpret_tensor(buf227, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf228, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf223, (1, 512), (512, 1), 0), reinterpret_tensor(primals_90, (512, 256), (256, 1), 0), reinterpret_tensor(buf217, (1, 256), (256, 1), 0), reinterpret_tensor(primals_87, (256, 512), (512, 1), 0), reinterpret_tensor(buf210, (1, 512), (512, 1), 0), reinterpret_tensor(primals_84, (512, 256), (256, 1), 0), reinterpret_tensor(buf204, (1, 256), (256, 1), 0), reinterpret_tensor(primals_81, (256, 512), (512, 1), 0), reinterpret_tensor(buf198, (64, 196, 49), (9604, 1, 196), 0), reinterpret_tensor(buf199, (64, 64, 196), (12544, 1, 64), 0), buf561, reinterpret_tensor(buf192, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf193, (64, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf188, (1, 128), (128, 1), 0), reinterpret_tensor(primals_78, (128, 128), (128, 1), 0), reinterpret_tensor(buf182, (1, 640), (640, 1), 0), reinterpret_tensor(primals_75, (640, 128), (128, 1), 0), reinterpret_tensor(buf175, (1, 128), (128, 1), 0), reinterpret_tensor(primals_72, (128, 256), (256, 1), 0), reinterpret_tensor(buf168, (1, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 128), (128, 1), 0), reinterpret_tensor(buf161, (1, 128), (128, 1), 0), reinterpret_tensor(primals_66, (128, 128), (128, 1), 0), reinterpret_tensor(buf155, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf156, (32, 32, 196), (6272, 1, 32), 0), buf562, reinterpret_tensor(buf149, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf150, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf145, (1, 256), (256, 1), 0), reinterpret_tensor(primals_63, (256, 128), (128, 1), 0), reinterpret_tensor(buf138, (1, 128), (128, 1), 0), reinterpret_tensor(primals_60, (128, 256), (256, 1), 0), reinterpret_tensor(buf131, (1, 256), (256, 1), 0), reinterpret_tensor(primals_57, (256, 128), (128, 1), 0), reinterpret_tensor(buf124, (1, 128), (128, 1), 0), reinterpret_tensor(primals_54, (128, 128), (128, 1), 0), reinterpret_tensor(buf118, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf119, (32, 32, 196), (6272, 1, 32), 0), buf563, reinterpret_tensor(buf112, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf113, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf108, (1, 256), (256, 1), 0), reinterpret_tensor(primals_51, (256, 128), (128, 1), 0), reinterpret_tensor(buf101, (1, 128), (128, 1), 0), reinterpret_tensor(primals_48, (128, 256), (256, 1), 0), reinterpret_tensor(buf94, (1, 256), (256, 1), 0), reinterpret_tensor(primals_45, (256, 128), (128, 1), 0), reinterpret_tensor(buf87, (1, 128), (128, 1), 0), reinterpret_tensor(primals_42, (128, 128), (128, 1), 0), reinterpret_tensor(buf81, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf82, (32, 32, 196), (6272, 1, 32), 0), buf564, reinterpret_tensor(buf75, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf76, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf71, (1, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 128), (128, 1), 0), reinterpret_tensor(buf64, (1, 128), (128, 1), 0), reinterpret_tensor(primals_36, (128, 256), (256, 1), 0), reinterpret_tensor(buf57, (1, 256), (256, 1), 0), reinterpret_tensor(primals_33, (256, 128), (128, 1), 0), reinterpret_tensor(buf50, (1, 128), (128, 1), 0), reinterpret_tensor(primals_30, (128, 128), (128, 1), 0), reinterpret_tensor(buf44, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf45, (32, 32, 196), (6272, 1, 32), 0), buf565, reinterpret_tensor(buf38, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf39, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf34, (1, 256), (256, 1), 0), reinterpret_tensor(primals_27, (256, 128), (128, 1), 0), reinterpret_tensor(buf27, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf20, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf6, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((8, 196), (196, 1), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((16, 49), (49, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((640, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_90 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_96 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_102 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_108 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((1280, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((384, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_180 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_204 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_210 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_211 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_212 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_213 = rand_strided((49, 196), (196, 1), device='cpu', dtype=torch.int64)
    primals_214 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_215 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_216 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_217 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_218 = rand_strided((16, 49), (49, 1), device='cpu', dtype=torch.int64)
    primals_219 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_220 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_221 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_222 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    primals_223 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_226 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_229 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_232 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_235 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_238 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_241 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_244 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_247 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_250 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_253 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_256 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_259 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_262 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_265 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_268 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_271 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_274 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_277 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_280 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_283 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_286 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_289 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_292 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_295 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_298 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_301 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_304 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_307 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_310 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_313 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_316 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_319 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_322 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_325 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_328 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_331 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_334 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_337 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_340 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_343 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_346 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_349 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_352 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_355 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_358 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_361 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_364 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_367 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_370 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_373 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_376 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_379 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_382 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_385 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_388 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_391 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_392 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_393 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_394 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_395 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_396 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_397 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_398 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_399 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_400 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_401 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_402 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_403 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_404 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_405 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_406 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_407 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_408 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_409 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_410 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_411 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_412 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_413 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    primals_414 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_415 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
