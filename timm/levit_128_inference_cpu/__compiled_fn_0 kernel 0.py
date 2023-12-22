
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr1[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                            out_ptr1[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_1 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)), static_cast<long>(16L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (144L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (16L*x2) + (144L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        #pragma omp single
        {
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
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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


cpp_fused__native_batch_norm_legit_no_training_clone_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38416L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 196);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 196L), "index out of bounds: 0 <= tmp3 < 196L")
                    auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (196L*x0))];
                    out_ptr0[static_cast<long>(x1 + (38416L*x0))] = tmp4;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_7 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38416L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 196);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 196L), "index out of bounds: 0 <= tmp3 < 196L")
                    auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (196L*x0))];
                    out_ptr0[static_cast<long>(x1 + (38416L*x0))] = tmp4;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_13 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38416L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 196);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 196L), "index out of bounds: 0 <= tmp3 < 196L")
                    auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (196L*x0))];
                    out_ptr0[static_cast<long>(x1 + (38416L*x0))] = tmp4;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_19 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_clone_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (3136L*x1) + (12544L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (256L*x3) + (50176L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(38416L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x1)];
                    auto tmp1 = decltype(tmp0)(tmp0 + 196);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 196L), "index out of bounds: 0 <= tmp3 < 196L")
                    auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (196L*x0))];
                    out_ptr0[static_cast<long>(x1 + (38416L*x0))] = tmp4;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (784L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (153664L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (784L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (153664L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (6272L*x1) + (25088L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_25 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((32L*x1) + (6272L*(c10::div_floor_integer((x2 + x2_inner), 32L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(32L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_add_clone_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_clone_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(1e-05);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 + tmp6;
                    auto tmp8 = tmp7.sqrt();
                    auto tmp9 = tmp8.reciprocal();
                    auto tmp10 = static_cast<float>(1.0);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp13 = tmp3 * tmp12;
                    auto tmp15 = tmp13 * tmp14;
                    auto tmp17 = tmp15 + tmp16;
                    auto tmp18 = tmp0 + tmp17;
                    tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                tmp0.store(out_ptr0 + static_cast<long>(x0));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (3584L*x0)));
                    tmp0.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_clone_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (128L*x2) + (6272L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (16L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                            at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x2 + (80L*x1) + (640L*x3) + (125440L*x0)), static_cast<long>(640L), tmp0, 8);
                            for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                                auto tmp2 = in_ptr6[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp5 = in_ptr7[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp14 = in_ptr8[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp17 = in_ptr9[static_cast<long>(x2 + x2_inner + (80L*x1))];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = static_cast<float>(1e-05);
                                auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                                auto tmp8 = std::sqrt(tmp7);
                                auto tmp9 = 1 / tmp8;
                                auto tmp10 = static_cast<float>(1.0);
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = at::vec::Vectorized<float>(tmp11);
                                auto tmp13 = tmp4 * tmp12;
                                auto tmp15 = at::vec::Vectorized<float>(tmp14);
                                auto tmp16 = tmp13 * tmp15;
                                auto tmp18 = at::vec::Vectorized<float>(tmp17);
                                auto tmp19 = tmp16 + tmp18;
                                tmp19.store(out_ptr1 + static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (25088L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(192L); x3<static_cast<long>(196L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (80L*x1) + (640L*x3) + (125440L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (80L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (80L*x1)));
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
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (196L*x2) + (196L*x2_inner) + (3136L*x1) + (25088L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(9604L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 196);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 196L), "index out of bounds: 0 <= tmp3 < 196L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (196L*x0))];
                out_ptr0[static_cast<long>(x1 + (9604L*x0))] = tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (196L*x1) + (76832L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (196L*x1) + (76832L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (392L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (76832L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (196L*x1) + (76832L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (76832L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (196L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (196L*x1) + (76832L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (196L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(192L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (196L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(16L + x3 + (80L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (12544L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_32 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (3136L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (25088L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (25088L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 49);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 49L), "index out of bounds: 0 <= tmp3 < 49L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (49L*x0))];
                out_ptr0[static_cast<long>(x1 + (2401L*x0))] = tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (392L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                                tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 49);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 49L), "index out of bounds: 0 <= tmp3 < 49L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (49L*x0))];
                out_ptr0[static_cast<long>(x1 + (2401L*x0))] = tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (392L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                                tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 49);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 49L), "index out of bounds: 0 <= tmp3 < 49L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (49L*x0))];
                out_ptr0[static_cast<long>(x1 + (2401L*x0))] = tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (392L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                                tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (784L*x1) + (6272L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(16L + x2 + (64L*x1) + (512L*x3) + (25088L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(16L + x2 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(16L + x2 + (64L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (6272L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2401L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 49);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 49L), "index out of bounds: 0 <= tmp3 < 49L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (49L*x0))];
                out_ptr0[static_cast<long>(x1 + (2401L*x0))] = tmp4;
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    {
                        #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                        float tmp_acc0 = -std::numeric_limits<float>::infinity();
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 * tmp2;
                            auto tmp5 = tmp3 + tmp4;
                            tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                        }
                        #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                        for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = in_ptr2[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                            auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                            auto tmp1 = static_cast<float>(0.25);
                            auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                            auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                            tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                        }
                        tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                        out_ptr1[static_cast<long>(x1 + (392L*x0))] = static_cast<float>(tmp_acc0);
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                        auto tmp6 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 - tmp7;
                        auto tmp9 = tmp8.exp();
                        tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (19208L*x0)));
                    }
                    #pragma omp simd simdlen(4) 
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                        auto tmp5 = out_ptr1[static_cast<long>(x1 + (392L*x0))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                        auto tmp7 = std::exp(tmp6);
                        in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (19208L*x0))] = tmp7;
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
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                    auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                                tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (1568L*x1) + (12544L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (12544L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200704L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
                tmp11.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_add_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(4L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (512L*x2) + (3584L*x1) + (12544L*x0)));
                        tmp0.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (1024L*x1) + (4096L*x0)));
                    }
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
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
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
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(1L))
                {
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x3 + (16L*x1) + (256L*x2) + (4096L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (16L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (16L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (4096L*x0)));
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
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x2 + (80L*x1) + (1280L*x3) + (62720L*x0)), static_cast<long>(1280L), tmp0, 8);
                        for (long x2_inner = 0; x2_inner < 8; x2_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x2_inner));
                            auto tmp2 = in_ptr6[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp5 = in_ptr7[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp14 = in_ptr8[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp17 = in_ptr9[static_cast<long>(x2 + x2_inner + (80L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (12544L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(48L); x3<static_cast<long>(49L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (80L*x1) + (1280L*x3) + (62720L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2 + (80L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (80L*x1)));
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x2_inner = 0; x2_inner < 8; x2_inner++) out_ptr1[static_cast<long>(x3 + (49L*x2) + (49L*x2_inner) + (784L*x1) + (12544L*x0))] = tmpbuf[x2_inner]; }
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 49);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 49L), "index out of bounds: 0 <= tmp3 < 49L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (49L*x0))];
                out_ptr0[static_cast<long>(x1 + (784L*x0))] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (49L*x1) + (12544L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                    }
                    #pragma omp simd simdlen(4)  reduction(max:tmp_acc0)
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr2[static_cast<long>(x2 + (49L*x1) + (12544L*x0))];
                        auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                        auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                        tmp_acc0 = max_propagate_nan(tmp_acc0, tmp4);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x1 + (256L*x0))] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (12544L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1)));
                    auto tmp6 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 - tmp7;
                    auto tmp9 = tmp8.exp();
                    tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (49L*x1) + (12544L*x0)));
                }
                #pragma omp simd simdlen(4) 
                for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (12544L*x0))];
                    auto tmp3 = out_ptr0[static_cast<long>(x2 + (49L*x1))];
                    auto tmp5 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp4 = decltype(tmp2)(tmp2 + tmp3);
                    auto tmp6 = decltype(tmp4)(tmp4 - tmp5);
                    auto tmp7 = std::exp(tmp6);
                    in_out_ptr0[static_cast<long>(x2 + (49L*x1) + (12544L*x0))] = tmp7;
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
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                tmp3.store(in_out_ptr0 + static_cast<long>(x1 + (49L*x0)));
            }
            #pragma omp simd simdlen(4) 
            for(long x1=static_cast<long>(48L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr0[static_cast<long>(x1 + (49L*x0))];
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(16L + x3 + (80L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(16L + x3 + (80L*x1)));
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
                            tmp16.store(out_ptr3 + static_cast<long>(x3 + (64L*x2) + (3136L*x1) + (50176L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = in_ptr0[static_cast<long>((64L*x1) + (1024L*(c10::div_floor_integer((x2 + x2_inner), 64L))) + (16384L*x0) + (static_cast<long>((x2 + x2_inner)) % static_cast<long>(64L)))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
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
                        tmp11.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (16384L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 16);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 16L), "index out of bounds: 0 <= tmp3 < 16L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (16L*x0))];
                out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                    auto tmp6 = out_ptr1[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 - tmp7;
                    auto tmp9 = tmp8.exp();
                    tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
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
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                        tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_71 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 16);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 16L), "index out of bounds: 0 <= tmp3 < 16L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (16L*x0))];
                out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                    auto tmp6 = out_ptr1[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 - tmp7;
                    auto tmp9 = tmp8.exp();
                    tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
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
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                        tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 16);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 16L), "index out of bounds: 0 <= tmp3 < 16L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (16L*x0))];
                out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                    auto tmp6 = out_ptr1[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 - tmp7;
                    auto tmp9 = tmp8.exp();
                    tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
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
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                        tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_82 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
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
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_clone_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1)
{
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
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x3 + (64L*x1)));
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
                        tmp16.store(out_ptr0 + static_cast<long>(x3 + (16L*x2) + (256L*x1) + (3072L*x0)));
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
                            auto tmp2 = in_ptr1[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp5 = in_ptr2[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp14 = in_ptr3[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp17 = in_ptr4[static_cast<long>(16L + x2 + x2_inner + (64L*x1))];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(1e-05);
                            auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                            auto tmp8 = std::sqrt(tmp7);
                            auto tmp9 = 1 / tmp8;
                            auto tmp10 = static_cast<float>(1.0);
                            auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp4 * tmp12;
                            auto tmp15 = at::vec::Vectorized<float>(tmp14);
                            auto tmp16 = tmp13 * tmp15;
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = tmp16 + tmp18;
                            tmp19.store(out_ptr1 + static_cast<long>(x3 + (16L*x2) + (16L*x2_inner) + (256L*x1) + (3072L*x0)));
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
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto in_ptr2 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(12L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x1)];
                auto tmp1 = decltype(tmp0)(tmp0 + 16);
                auto tmp2 = tmp0 < 0;
                auto tmp3 = tmp2 ? tmp1 : tmp0;
                TORCH_CHECK((0 <= tmp3) & (tmp3 < 16L), "index out of bounds: 0 <= tmp3 < 16L")
                auto tmp4 = in_ptr1[static_cast<long>(tmp3 + (16L*x0))];
                out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp4;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(max:at::vec::Vectorized<float>:omp_out = at::vec::maximum(omp_out, omp_in)) initializer(omp_priv={at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity())})
                    float tmp_acc0 = -std::numeric_limits<float>::infinity();
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(-std::numeric_limits<float>::infinity());
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                        auto tmp1 = static_cast<float>(0.25);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = tmp0 * tmp2;
                        auto tmp5 = tmp3 + tmp4;
                        tmp_acc0_vec = at::vec::maximum(tmp_acc0_vec, tmp5);
                    }
                    tmp_acc0 = max_propagate_nan(tmp_acc0, at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return at::vec::maximum(x, y); }, tmp_acc0_vec));
                    out_ptr1[static_cast<long>(x1 + (192L*x0))] = static_cast<float>(tmp_acc0);
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(16L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (16L*x1)));
                    auto tmp6 = out_ptr1[static_cast<long>(x1 + (192L*x0))];
                    auto tmp1 = static_cast<float>(0.25);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 * tmp2;
                    auto tmp5 = tmp3 + tmp4;
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 - tmp7;
                    auto tmp9 = tmp8.exp();
                    tmp9.store(in_out_ptr0 + static_cast<long>(x2 + (16L*x1) + (3072L*x0)));
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
                out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
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
                auto tmp1 = out_ptr2[static_cast<long>(x0)];
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
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(32L + x3 + (64L*x1)));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(32L + x3 + (64L*x1)));
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
                        tmp16.store(out_ptr3 + static_cast<long>(x3 + (32L*x2) + (512L*x1) + (6144L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_hardswish_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    tmp11.store(out_ptr0 + static_cast<long>(x2 + (384L*x1) + (6144L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_88 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp3 = tmp1 - tmp2;
                auto tmp5 = static_cast<float>(1e-05);
                auto tmp6 = at::vec::Vectorized<float>(tmp5);
                auto tmp7 = tmp4 + tmp6;
                auto tmp8 = tmp7.sqrt();
                auto tmp9 = tmp8.reciprocal();
                auto tmp10 = static_cast<float>(1.0);
                auto tmp11 = at::vec::Vectorized<float>(tmp10);
                auto tmp12 = tmp9 * tmp11;
                auto tmp13 = tmp3 * tmp12;
                auto tmp15 = tmp13 * tmp14;
                auto tmp17 = tmp15 + tmp16;
                auto tmp18 = tmp0 + tmp17;
                tmp18.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_hardswish_89 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(98304L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
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
            tmp11.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (6144L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (384L*x2) + (6144L*x0)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp5 = static_cast<float>(1e-05);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = tmp4 + tmp6;
                        auto tmp8 = tmp7.sqrt();
                        auto tmp9 = tmp8.reciprocal();
                        auto tmp10 = static_cast<float>(1.0);
                        auto tmp11 = at::vec::Vectorized<float>(tmp10);
                        auto tmp12 = tmp9 * tmp11;
                        auto tmp13 = tmp3 * tmp12;
                        auto tmp15 = tmp13 * tmp14;
                        auto tmp17 = tmp15 + tmp16;
                        auto tmp18 = tmp0 + tmp17;
                        tmp_acc0_vec = tmp_acc0_vec + tmp18;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
                auto tmp1 = static_cast<float>(16.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = tmp0 / tmp2;
                auto tmp5 = tmp3 - tmp4;
                auto tmp7 = static_cast<float>(1e-05);
                auto tmp8 = at::vec::Vectorized<float>(tmp7);
                auto tmp9 = tmp6 + tmp8;
                auto tmp10 = tmp9.sqrt();
                auto tmp11 = tmp10.reciprocal();
                auto tmp12 = static_cast<float>(1.0);
                auto tmp13 = at::vec::Vectorized<float>(tmp12);
                auto tmp14 = tmp11 * tmp13;
                auto tmp15 = tmp5 * tmp14;
                auto tmp17 = tmp15 * tmp16;
                auto tmp19 = tmp17 + tmp18;
                auto tmp21 = tmp3 - tmp20;
                auto tmp23 = tmp22 + tmp8;
                auto tmp24 = tmp23.sqrt();
                auto tmp25 = tmp24.reciprocal();
                auto tmp26 = tmp25 * tmp13;
                auto tmp27 = tmp21 * tmp26;
                auto tmp29 = tmp27 * tmp28;
                auto tmp31 = tmp29 + tmp30;
                tmp19.store(out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                tmp31.store(out_ptr2 + static_cast<long>(x1 + (384L*x0)));
            }
        }
    }
}
''')


cpp_fused_add_div_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
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
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 196), (196, 1))
    assert_size_stride(arg1_1, (4, 196), (196, 1))
    assert_size_stride(arg2_1, (4, 196), (196, 1))
    assert_size_stride(arg3_1, (4, 196), (196, 1))
    assert_size_stride(arg4_1, (8, 196), (196, 1))
    assert_size_stride(arg5_1, (8, 49), (49, 1))
    assert_size_stride(arg6_1, (8, 49), (49, 1))
    assert_size_stride(arg7_1, (8, 49), (49, 1))
    assert_size_stride(arg8_1, (8, 49), (49, 1))
    assert_size_stride(arg9_1, (16, 49), (49, 1))
    assert_size_stride(arg10_1, (12, 16), (16, 1))
    assert_size_stride(arg11_1, (12, 16), (16, 1))
    assert_size_stride(arg12_1, (12, 16), (16, 1))
    assert_size_stride(arg13_1, (12, 16), (16, 1))
    assert_size_stride(arg14_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg15_1, (16, ), (1, ))
    assert_size_stride(arg16_1, (16, ), (1, ))
    assert_size_stride(arg17_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg18_1, (32, ), (1, ))
    assert_size_stride(arg19_1, (32, ), (1, ))
    assert_size_stride(arg20_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (256, 128), (128, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (128, 128), (128, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (256, 128), (128, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (128, 256), (256, 1))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (256, 128), (128, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (128, 128), (128, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (256, 128), (128, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (128, 256), (256, 1))
    assert_size_stride(arg48_1, (128, ), (1, ))
    assert_size_stride(arg49_1, (128, ), (1, ))
    assert_size_stride(arg50_1, (256, 128), (128, 1))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (128, 128), (128, 1))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (256, 128), (128, 1))
    assert_size_stride(arg57_1, (256, ), (1, ))
    assert_size_stride(arg58_1, (256, ), (1, ))
    assert_size_stride(arg59_1, (128, 256), (256, 1))
    assert_size_stride(arg60_1, (128, ), (1, ))
    assert_size_stride(arg61_1, (128, ), (1, ))
    assert_size_stride(arg62_1, (256, 128), (128, 1))
    assert_size_stride(arg63_1, (256, ), (1, ))
    assert_size_stride(arg64_1, (256, ), (1, ))
    assert_size_stride(arg65_1, (128, 128), (128, 1))
    assert_size_stride(arg66_1, (128, ), (1, ))
    assert_size_stride(arg67_1, (128, ), (1, ))
    assert_size_stride(arg68_1, (256, 128), (128, 1))
    assert_size_stride(arg69_1, (256, ), (1, ))
    assert_size_stride(arg70_1, (256, ), (1, ))
    assert_size_stride(arg71_1, (128, 256), (256, 1))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (640, 128), (128, 1))
    assert_size_stride(arg75_1, (640, ), (1, ))
    assert_size_stride(arg76_1, (640, ), (1, ))
    assert_size_stride(arg77_1, (128, 128), (128, 1))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (256, 512), (512, 1))
    assert_size_stride(arg81_1, (256, ), (1, ))
    assert_size_stride(arg82_1, (256, ), (1, ))
    assert_size_stride(arg83_1, (512, 256), (256, 1))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (256, 512), (512, 1))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (256, ), (1, ))
    assert_size_stride(arg89_1, (512, 256), (256, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (256, 256), (256, 1))
    assert_size_stride(arg93_1, (256, ), (1, ))
    assert_size_stride(arg94_1, (256, ), (1, ))
    assert_size_stride(arg95_1, (512, 256), (256, 1))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (512, ), (1, ))
    assert_size_stride(arg98_1, (256, 512), (512, 1))
    assert_size_stride(arg99_1, (256, ), (1, ))
    assert_size_stride(arg100_1, (256, ), (1, ))
    assert_size_stride(arg101_1, (512, 256), (256, 1))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (512, ), (1, ))
    assert_size_stride(arg104_1, (256, 256), (256, 1))
    assert_size_stride(arg105_1, (256, ), (1, ))
    assert_size_stride(arg106_1, (256, ), (1, ))
    assert_size_stride(arg107_1, (512, 256), (256, 1))
    assert_size_stride(arg108_1, (512, ), (1, ))
    assert_size_stride(arg109_1, (512, ), (1, ))
    assert_size_stride(arg110_1, (256, 512), (512, 1))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (256, ), (1, ))
    assert_size_stride(arg113_1, (512, 256), (256, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (256, 256), (256, 1))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (512, 256), (256, 1))
    assert_size_stride(arg120_1, (512, ), (1, ))
    assert_size_stride(arg121_1, (512, ), (1, ))
    assert_size_stride(arg122_1, (256, 512), (512, 1))
    assert_size_stride(arg123_1, (256, ), (1, ))
    assert_size_stride(arg124_1, (256, ), (1, ))
    assert_size_stride(arg125_1, (512, 256), (256, 1))
    assert_size_stride(arg126_1, (512, ), (1, ))
    assert_size_stride(arg127_1, (512, ), (1, ))
    assert_size_stride(arg128_1, (256, 256), (256, 1))
    assert_size_stride(arg129_1, (256, ), (1, ))
    assert_size_stride(arg130_1, (256, ), (1, ))
    assert_size_stride(arg131_1, (512, 256), (256, 1))
    assert_size_stride(arg132_1, (512, ), (1, ))
    assert_size_stride(arg133_1, (512, ), (1, ))
    assert_size_stride(arg134_1, (256, 512), (512, 1))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (1280, 256), (256, 1))
    assert_size_stride(arg138_1, (1280, ), (1, ))
    assert_size_stride(arg139_1, (1280, ), (1, ))
    assert_size_stride(arg140_1, (256, 256), (256, 1))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (384, 1024), (1024, 1))
    assert_size_stride(arg144_1, (384, ), (1, ))
    assert_size_stride(arg145_1, (384, ), (1, ))
    assert_size_stride(arg146_1, (768, 384), (384, 1))
    assert_size_stride(arg147_1, (768, ), (1, ))
    assert_size_stride(arg148_1, (768, ), (1, ))
    assert_size_stride(arg149_1, (384, 768), (768, 1))
    assert_size_stride(arg150_1, (384, ), (1, ))
    assert_size_stride(arg151_1, (384, ), (1, ))
    assert_size_stride(arg152_1, (768, 384), (384, 1))
    assert_size_stride(arg153_1, (768, ), (1, ))
    assert_size_stride(arg154_1, (768, ), (1, ))
    assert_size_stride(arg155_1, (384, 384), (384, 1))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (768, 384), (384, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (384, 768), (768, 1))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (768, 384), (384, 1))
    assert_size_stride(arg165_1, (768, ), (1, ))
    assert_size_stride(arg166_1, (768, ), (1, ))
    assert_size_stride(arg167_1, (384, 384), (384, 1))
    assert_size_stride(arg168_1, (384, ), (1, ))
    assert_size_stride(arg169_1, (384, ), (1, ))
    assert_size_stride(arg170_1, (768, 384), (384, 1))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (384, 768), (768, 1))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (768, 384), (384, 1))
    assert_size_stride(arg177_1, (768, ), (1, ))
    assert_size_stride(arg178_1, (768, ), (1, ))
    assert_size_stride(arg179_1, (384, 384), (384, 1))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (768, 384), (384, 1))
    assert_size_stride(arg183_1, (768, ), (1, ))
    assert_size_stride(arg184_1, (768, ), (1, ))
    assert_size_stride(arg185_1, (384, 768), (768, 1))
    assert_size_stride(arg186_1, (384, ), (1, ))
    assert_size_stride(arg187_1, (384, ), (1, ))
    assert_size_stride(arg188_1, (768, 384), (384, 1))
    assert_size_stride(arg189_1, (768, ), (1, ))
    assert_size_stride(arg190_1, (768, ), (1, ))
    assert_size_stride(arg191_1, (384, 384), (384, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (384, ), (1, ))
    assert_size_stride(arg194_1, (768, 384), (384, 1))
    assert_size_stride(arg195_1, (768, ), (1, ))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (384, 768), (768, 1))
    assert_size_stride(arg198_1, (384, ), (1, ))
    assert_size_stride(arg199_1, (384, ), (1, ))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (384, ), (1, ))
    assert_size_stride(arg202_1, (1000, 384), (384, 1))
    assert_size_stride(arg203_1, (1000, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (1000, 384), (384, 1))
    assert_size_stride(arg207_1, (1000, ), (1, ))
    assert_size_stride(arg208_1, (196, 196), (196, 1))
    assert_size_stride(arg209_1, (196, 196), (196, 1))
    assert_size_stride(arg210_1, (196, 196), (196, 1))
    assert_size_stride(arg211_1, (196, 196), (196, 1))
    assert_size_stride(arg212_1, (49, 196), (196, 1))
    assert_size_stride(arg213_1, (49, 49), (49, 1))
    assert_size_stride(arg214_1, (49, 49), (49, 1))
    assert_size_stride(arg215_1, (49, 49), (49, 1))
    assert_size_stride(arg216_1, (49, 49), (49, 1))
    assert_size_stride(arg217_1, (16, 49), (49, 1))
    assert_size_stride(arg218_1, (16, 16), (16, 1))
    assert_size_stride(arg219_1, (16, 16), (16, 1))
    assert_size_stride(arg220_1, (16, 16), (16, 1))
    assert_size_stride(arg221_1, (16, 16), (16, 1))
    assert_size_stride(arg222_1, (16, ), (1, ))
    assert_size_stride(arg223_1, (16, ), (1, ))
    assert_size_stride(arg224_1, (), ())
    assert_size_stride(arg225_1, (32, ), (1, ))
    assert_size_stride(arg226_1, (32, ), (1, ))
    assert_size_stride(arg227_1, (), ())
    assert_size_stride(arg228_1, (64, ), (1, ))
    assert_size_stride(arg229_1, (64, ), (1, ))
    assert_size_stride(arg230_1, (), ())
    assert_size_stride(arg231_1, (128, ), (1, ))
    assert_size_stride(arg232_1, (128, ), (1, ))
    assert_size_stride(arg233_1, (), ())
    assert_size_stride(arg234_1, (256, ), (1, ))
    assert_size_stride(arg235_1, (256, ), (1, ))
    assert_size_stride(arg236_1, (), ())
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (), ())
    assert_size_stride(arg240_1, (256, ), (1, ))
    assert_size_stride(arg241_1, (256, ), (1, ))
    assert_size_stride(arg242_1, (), ())
    assert_size_stride(arg243_1, (128, ), (1, ))
    assert_size_stride(arg244_1, (128, ), (1, ))
    assert_size_stride(arg245_1, (), ())
    assert_size_stride(arg246_1, (256, ), (1, ))
    assert_size_stride(arg247_1, (256, ), (1, ))
    assert_size_stride(arg248_1, (), ())
    assert_size_stride(arg249_1, (128, ), (1, ))
    assert_size_stride(arg250_1, (128, ), (1, ))
    assert_size_stride(arg251_1, (), ())
    assert_size_stride(arg252_1, (256, ), (1, ))
    assert_size_stride(arg253_1, (256, ), (1, ))
    assert_size_stride(arg254_1, (), ())
    assert_size_stride(arg255_1, (128, ), (1, ))
    assert_size_stride(arg256_1, (128, ), (1, ))
    assert_size_stride(arg257_1, (), ())
    assert_size_stride(arg258_1, (256, ), (1, ))
    assert_size_stride(arg259_1, (256, ), (1, ))
    assert_size_stride(arg260_1, (), ())
    assert_size_stride(arg261_1, (128, ), (1, ))
    assert_size_stride(arg262_1, (128, ), (1, ))
    assert_size_stride(arg263_1, (), ())
    assert_size_stride(arg264_1, (256, ), (1, ))
    assert_size_stride(arg265_1, (256, ), (1, ))
    assert_size_stride(arg266_1, (), ())
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (), ())
    assert_size_stride(arg270_1, (256, ), (1, ))
    assert_size_stride(arg271_1, (256, ), (1, ))
    assert_size_stride(arg272_1, (), ())
    assert_size_stride(arg273_1, (128, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (), ())
    assert_size_stride(arg276_1, (256, ), (1, ))
    assert_size_stride(arg277_1, (256, ), (1, ))
    assert_size_stride(arg278_1, (), ())
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (), ())
    assert_size_stride(arg282_1, (640, ), (1, ))
    assert_size_stride(arg283_1, (640, ), (1, ))
    assert_size_stride(arg284_1, (), ())
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (), ())
    assert_size_stride(arg288_1, (256, ), (1, ))
    assert_size_stride(arg289_1, (256, ), (1, ))
    assert_size_stride(arg290_1, (), ())
    assert_size_stride(arg291_1, (512, ), (1, ))
    assert_size_stride(arg292_1, (512, ), (1, ))
    assert_size_stride(arg293_1, (), ())
    assert_size_stride(arg294_1, (256, ), (1, ))
    assert_size_stride(arg295_1, (256, ), (1, ))
    assert_size_stride(arg296_1, (), ())
    assert_size_stride(arg297_1, (512, ), (1, ))
    assert_size_stride(arg298_1, (512, ), (1, ))
    assert_size_stride(arg299_1, (), ())
    assert_size_stride(arg300_1, (256, ), (1, ))
    assert_size_stride(arg301_1, (256, ), (1, ))
    assert_size_stride(arg302_1, (), ())
    assert_size_stride(arg303_1, (512, ), (1, ))
    assert_size_stride(arg304_1, (512, ), (1, ))
    assert_size_stride(arg305_1, (), ())
    assert_size_stride(arg306_1, (256, ), (1, ))
    assert_size_stride(arg307_1, (256, ), (1, ))
    assert_size_stride(arg308_1, (), ())
    assert_size_stride(arg309_1, (512, ), (1, ))
    assert_size_stride(arg310_1, (512, ), (1, ))
    assert_size_stride(arg311_1, (), ())
    assert_size_stride(arg312_1, (256, ), (1, ))
    assert_size_stride(arg313_1, (256, ), (1, ))
    assert_size_stride(arg314_1, (), ())
    assert_size_stride(arg315_1, (512, ), (1, ))
    assert_size_stride(arg316_1, (512, ), (1, ))
    assert_size_stride(arg317_1, (), ())
    assert_size_stride(arg318_1, (256, ), (1, ))
    assert_size_stride(arg319_1, (256, ), (1, ))
    assert_size_stride(arg320_1, (), ())
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (512, ), (1, ))
    assert_size_stride(arg323_1, (), ())
    assert_size_stride(arg324_1, (256, ), (1, ))
    assert_size_stride(arg325_1, (256, ), (1, ))
    assert_size_stride(arg326_1, (), ())
    assert_size_stride(arg327_1, (512, ), (1, ))
    assert_size_stride(arg328_1, (512, ), (1, ))
    assert_size_stride(arg329_1, (), ())
    assert_size_stride(arg330_1, (256, ), (1, ))
    assert_size_stride(arg331_1, (256, ), (1, ))
    assert_size_stride(arg332_1, (), ())
    assert_size_stride(arg333_1, (512, ), (1, ))
    assert_size_stride(arg334_1, (512, ), (1, ))
    assert_size_stride(arg335_1, (), ())
    assert_size_stride(arg336_1, (256, ), (1, ))
    assert_size_stride(arg337_1, (256, ), (1, ))
    assert_size_stride(arg338_1, (), ())
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (512, ), (1, ))
    assert_size_stride(arg341_1, (), ())
    assert_size_stride(arg342_1, (256, ), (1, ))
    assert_size_stride(arg343_1, (256, ), (1, ))
    assert_size_stride(arg344_1, (), ())
    assert_size_stride(arg345_1, (1280, ), (1, ))
    assert_size_stride(arg346_1, (1280, ), (1, ))
    assert_size_stride(arg347_1, (), ())
    assert_size_stride(arg348_1, (256, ), (1, ))
    assert_size_stride(arg349_1, (256, ), (1, ))
    assert_size_stride(arg350_1, (), ())
    assert_size_stride(arg351_1, (384, ), (1, ))
    assert_size_stride(arg352_1, (384, ), (1, ))
    assert_size_stride(arg353_1, (), ())
    assert_size_stride(arg354_1, (768, ), (1, ))
    assert_size_stride(arg355_1, (768, ), (1, ))
    assert_size_stride(arg356_1, (), ())
    assert_size_stride(arg357_1, (384, ), (1, ))
    assert_size_stride(arg358_1, (384, ), (1, ))
    assert_size_stride(arg359_1, (), ())
    assert_size_stride(arg360_1, (768, ), (1, ))
    assert_size_stride(arg361_1, (768, ), (1, ))
    assert_size_stride(arg362_1, (), ())
    assert_size_stride(arg363_1, (384, ), (1, ))
    assert_size_stride(arg364_1, (384, ), (1, ))
    assert_size_stride(arg365_1, (), ())
    assert_size_stride(arg366_1, (768, ), (1, ))
    assert_size_stride(arg367_1, (768, ), (1, ))
    assert_size_stride(arg368_1, (), ())
    assert_size_stride(arg369_1, (384, ), (1, ))
    assert_size_stride(arg370_1, (384, ), (1, ))
    assert_size_stride(arg371_1, (), ())
    assert_size_stride(arg372_1, (768, ), (1, ))
    assert_size_stride(arg373_1, (768, ), (1, ))
    assert_size_stride(arg374_1, (), ())
    assert_size_stride(arg375_1, (384, ), (1, ))
    assert_size_stride(arg376_1, (384, ), (1, ))
    assert_size_stride(arg377_1, (), ())
    assert_size_stride(arg378_1, (768, ), (1, ))
    assert_size_stride(arg379_1, (768, ), (1, ))
    assert_size_stride(arg380_1, (), ())
    assert_size_stride(arg381_1, (384, ), (1, ))
    assert_size_stride(arg382_1, (384, ), (1, ))
    assert_size_stride(arg383_1, (), ())
    assert_size_stride(arg384_1, (768, ), (1, ))
    assert_size_stride(arg385_1, (768, ), (1, ))
    assert_size_stride(arg386_1, (), ())
    assert_size_stride(arg387_1, (384, ), (1, ))
    assert_size_stride(arg388_1, (384, ), (1, ))
    assert_size_stride(arg389_1, (), ())
    assert_size_stride(arg390_1, (768, ), (1, ))
    assert_size_stride(arg391_1, (768, ), (1, ))
    assert_size_stride(arg392_1, (), ())
    assert_size_stride(arg393_1, (384, ), (1, ))
    assert_size_stride(arg394_1, (384, ), (1, ))
    assert_size_stride(arg395_1, (), ())
    assert_size_stride(arg396_1, (768, ), (1, ))
    assert_size_stride(arg397_1, (768, ), (1, ))
    assert_size_stride(arg398_1, (), ())
    assert_size_stride(arg399_1, (384, ), (1, ))
    assert_size_stride(arg400_1, (384, ), (1, ))
    assert_size_stride(arg401_1, (), ())
    assert_size_stride(arg402_1, (768, ), (1, ))
    assert_size_stride(arg403_1, (768, ), (1, ))
    assert_size_stride(arg404_1, (), ())
    assert_size_stride(arg405_1, (384, ), (1, ))
    assert_size_stride(arg406_1, (384, ), (1, ))
    assert_size_stride(arg407_1, (), ())
    assert_size_stride(arg408_1, (384, ), (1, ))
    assert_size_stride(arg409_1, (384, ), (1, ))
    assert_size_stride(arg410_1, (), ())
    assert_size_stride(arg411_1, (384, ), (1, ))
    assert_size_stride(arg412_1, (384, ), (1, ))
    assert_size_stride(arg413_1, (), ())
    assert_size_stride(arg414_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg414_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg14_1
    del arg414_1
    # Source Nodes: [l__mod___stem_conv1_linear], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 16, 112, 112), (200704, 1, 1792, 16))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = buf3; del buf3  # reuse
    buf5 = empty_strided((32, 16, 3, 3), (144, 1, 48, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_1(c_void_p(buf4.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf5.data_ptr()))
    del arg15_1
    del arg16_1
    del arg17_1
    del arg222_1
    del arg223_1
    # Source Nodes: [l__mod___stem_act1, l__mod___stem_conv2_linear], Original ATen: [aten.convolution, aten.hardswish]
    buf6 = extern_kernels.convolution(buf4, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 32, 56, 56), (100352, 1, 1792, 32))
    del buf4
    del buf5
    buf7 = buf6; del buf6  # reuse
    buf8 = buf7; del buf7  # reuse
    buf9 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_2(c_void_p(buf8.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg18_1
    del arg19_1
    del arg20_1
    del arg225_1
    del arg226_1
    # Source Nodes: [l__mod___stem_act2, l__mod___stem_conv3_linear], Original ATen: [aten.convolution, aten.hardswish]
    buf10 = extern_kernels.convolution(buf8, buf9, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (8, 64, 28, 28), (50176, 1, 1792, 64))
    del buf9
    buf11 = buf10; del buf10  # reuse
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_hardswish_3(c_void_p(buf12.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg21_1
    del arg228_1
    del arg229_1
    del arg22_1
    del arg23_1
    # Source Nodes: [l__mod___stem_act3, l__mod___stem_conv4_linear], Original ATen: [aten.convolution, aten.hardswish]
    buf14 = extern_kernels.convolution(buf12, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 128, 14, 14), (25088, 1, 1792, 128))
    del buf13
    buf15 = buf14; del buf14  # reuse
    buf16 = empty((8, 196, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_clone_4(c_void_p(buf15.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg231_1
    del arg232_1
    del arg24_1
    del arg25_1
    buf17 = reinterpret_tensor(buf12, (1568, 256), (256, 1), 0); del buf12  # reuse
    # Source Nodes: [x_3], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf16, (1568, 128), (128, 1), 0), reinterpret_tensor(arg26_1, (128, 256), (1, 128), 0), out=buf17)
    del arg26_1
    buf18 = empty((8, 4, 196, 16), device='cpu', dtype=torch.float32)
    buf19 = empty((8, 4, 16, 196), device='cpu', dtype=torch.float32)
    cpp_fused_clone_5(c_void_p(buf17.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    buf20 = empty((32, 196, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf18, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf19, (32, 16, 196), (3136, 196, 1), 0), out=buf20)
    buf21 = empty((4, 196, 196), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf23 = reinterpret_tensor(buf20, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf20  # reuse
    buf24 = empty_strided((8, 4, 196, 1), (784, 196, 1, 6272), device='cpu', dtype=torch.float32)
    buf25 = buf23; del buf23  # reuse
    buf26 = reinterpret_tensor(buf16, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf16  # reuse
    cpp_fused__softmax_add_clone_index_mul_6(c_void_p(buf25.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf26.data_ptr()))
    del arg0_1
    del arg208_1
    del arg234_1
    del arg235_1
    del arg27_1
    del arg28_1
    buf27 = empty((32, 196, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf25, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf26, (32, 196, 32), (6272, 32, 1), 0), out=buf27)
    buf28 = reinterpret_tensor(buf26, (8, 196, 128), (25088, 128, 1), 0); del buf26  # reuse
    cpp_fused_hardswish_7(c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()))
    buf29 = reinterpret_tensor(buf27, (1568, 128), (128, 1), 0); del buf27  # reuse
    # Source Nodes: [x_5], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf28, (1568, 128), (128, 1), 0), reinterpret_tensor(arg29_1, (128, 128), (1, 128), 0), out=buf29)
    del arg29_1
    buf30 = reinterpret_tensor(buf15, (8, 196, 128), (25088, 128, 1), 0); del buf15  # reuse
    buf31 = buf28; del buf28  # reuse
    cpp_fused_add_clone_8(c_void_p(buf30.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg237_1
    del arg238_1
    del arg30_1
    del arg31_1
    buf32 = buf17; del buf17  # reuse
    # Source Nodes: [x_8], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf31, (1568, 128), (128, 1), 0), reinterpret_tensor(arg32_1, (128, 256), (1, 128), 0), out=buf32)
    del arg32_1
    buf33 = buf32; del buf32  # reuse
    buf34 = reinterpret_tensor(buf33, (8, 196, 256), (50176, 256, 1), 0); del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_9(c_void_p(buf34.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg34_1.data_ptr()))
    del arg240_1
    del arg241_1
    del arg33_1
    del arg34_1
    buf35 = reinterpret_tensor(buf31, (1568, 128), (128, 1), 0); del buf31  # reuse
    # Source Nodes: [x_12], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf34, (1568, 256), (256, 1), 0), reinterpret_tensor(arg35_1, (256, 128), (1, 256), 0), out=buf35)
    del arg35_1
    buf36 = buf30; del buf30  # reuse
    buf37 = reinterpret_tensor(buf29, (8, 196, 128), (25088, 128, 1), 0); del buf29  # reuse
    cpp_fused_add_clone_10(c_void_p(buf36.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg243_1
    del arg244_1
    del arg36_1
    del arg37_1
    buf38 = reinterpret_tensor(buf34, (1568, 256), (256, 1), 0); del buf34  # reuse
    # Source Nodes: [x_15], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf37, (1568, 128), (128, 1), 0), reinterpret_tensor(arg38_1, (128, 256), (1, 128), 0), out=buf38)
    del arg38_1
    buf39 = reinterpret_tensor(buf19, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf19  # reuse
    buf40 = reinterpret_tensor(buf18, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf18  # reuse
    cpp_fused_clone_11(c_void_p(buf38.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()))
    buf41 = reinterpret_tensor(buf25, (32, 196, 196), (38416, 196, 1), 0); del buf25  # reuse
    # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf39, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf40, (32, 16, 196), (3136, 196, 1), 0), out=buf41)
    buf42 = empty((4, 196, 196), device='cpu', dtype=torch.float32)
    buf43 = buf24; del buf24  # reuse
    buf44 = reinterpret_tensor(buf41, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf41  # reuse
    buf45 = buf22; del buf22  # reuse
    buf46 = buf44; del buf44  # reuse
    buf47 = reinterpret_tensor(buf37, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf37  # reuse
    cpp_fused__softmax_add_clone_index_mul_12(c_void_p(buf46.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf47.data_ptr()))
    del arg1_1
    del arg209_1
    del arg246_1
    del arg247_1
    del arg39_1
    del arg40_1
    buf48 = reinterpret_tensor(buf35, (32, 196, 32), (6272, 32, 1), 0); del buf35  # reuse
    # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf46, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf47, (32, 196, 32), (6272, 32, 1), 0), out=buf48)
    buf49 = reinterpret_tensor(buf47, (8, 196, 128), (25088, 128, 1), 0); del buf47  # reuse
    cpp_fused_hardswish_13(c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()))
    buf50 = reinterpret_tensor(buf48, (1568, 128), (128, 1), 0); del buf48  # reuse
    # Source Nodes: [x_17], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf49, (1568, 128), (128, 1), 0), reinterpret_tensor(arg41_1, (128, 128), (1, 128), 0), out=buf50)
    del arg41_1
    buf51 = buf36; del buf36  # reuse
    buf52 = buf49; del buf49  # reuse
    cpp_fused_add_clone_14(c_void_p(buf51.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg249_1
    del arg250_1
    del arg42_1
    del arg43_1
    buf53 = buf38; del buf38  # reuse
    # Source Nodes: [x_20], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf52, (1568, 128), (128, 1), 0), reinterpret_tensor(arg44_1, (128, 256), (1, 128), 0), out=buf53)
    del arg44_1
    buf54 = buf53; del buf53  # reuse
    buf55 = reinterpret_tensor(buf54, (8, 196, 256), (50176, 256, 1), 0); del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_15(c_void_p(buf55.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg46_1.data_ptr()))
    del arg252_1
    del arg253_1
    del arg45_1
    del arg46_1
    buf56 = reinterpret_tensor(buf52, (1568, 128), (128, 1), 0); del buf52  # reuse
    # Source Nodes: [x_24], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf55, (1568, 256), (256, 1), 0), reinterpret_tensor(arg47_1, (256, 128), (1, 256), 0), out=buf56)
    del arg47_1
    buf57 = buf51; del buf51  # reuse
    buf58 = reinterpret_tensor(buf50, (8, 196, 128), (25088, 128, 1), 0); del buf50  # reuse
    cpp_fused_add_clone_16(c_void_p(buf57.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf58.data_ptr()))
    del arg255_1
    del arg256_1
    del arg48_1
    del arg49_1
    buf59 = reinterpret_tensor(buf55, (1568, 256), (256, 1), 0); del buf55  # reuse
    # Source Nodes: [x_27], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf58, (1568, 128), (128, 1), 0), reinterpret_tensor(arg50_1, (128, 256), (1, 128), 0), out=buf59)
    del arg50_1
    buf60 = reinterpret_tensor(buf40, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf40  # reuse
    buf61 = reinterpret_tensor(buf39, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf39  # reuse
    cpp_fused_clone_17(c_void_p(buf59.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()))
    buf62 = reinterpret_tensor(buf46, (32, 196, 196), (38416, 196, 1), 0); del buf46  # reuse
    # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf60, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf61, (32, 16, 196), (3136, 196, 1), 0), out=buf62)
    buf63 = empty((4, 196, 196), device='cpu', dtype=torch.float32)
    buf64 = buf45; del buf45  # reuse
    buf65 = reinterpret_tensor(buf62, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf62  # reuse
    buf66 = buf43; del buf43  # reuse
    buf67 = buf65; del buf65  # reuse
    buf68 = reinterpret_tensor(buf58, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf58  # reuse
    cpp_fused__softmax_add_clone_index_mul_18(c_void_p(buf67.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg210_1
    del arg258_1
    del arg259_1
    del arg2_1
    del arg51_1
    del arg52_1
    buf69 = reinterpret_tensor(buf56, (32, 196, 32), (6272, 32, 1), 0); del buf56  # reuse
    # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf67, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf68, (32, 196, 32), (6272, 32, 1), 0), out=buf69)
    buf70 = reinterpret_tensor(buf68, (8, 196, 128), (25088, 128, 1), 0); del buf68  # reuse
    cpp_fused_hardswish_19(c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    buf71 = reinterpret_tensor(buf69, (1568, 128), (128, 1), 0); del buf69  # reuse
    # Source Nodes: [x_29], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf70, (1568, 128), (128, 1), 0), reinterpret_tensor(arg53_1, (128, 128), (1, 128), 0), out=buf71)
    del arg53_1
    buf72 = buf57; del buf57  # reuse
    buf73 = buf70; del buf70  # reuse
    cpp_fused_add_clone_20(c_void_p(buf72.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg261_1
    del arg262_1
    del arg54_1
    del arg55_1
    buf74 = buf59; del buf59  # reuse
    # Source Nodes: [x_32], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf73, (1568, 128), (128, 1), 0), reinterpret_tensor(arg56_1, (128, 256), (1, 128), 0), out=buf74)
    del arg56_1
    buf75 = buf74; del buf74  # reuse
    buf76 = reinterpret_tensor(buf75, (8, 196, 256), (50176, 256, 1), 0); del buf75  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_21(c_void_p(buf76.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()))
    del arg264_1
    del arg265_1
    del arg57_1
    del arg58_1
    buf77 = reinterpret_tensor(buf73, (1568, 128), (128, 1), 0); del buf73  # reuse
    # Source Nodes: [x_36], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf76, (1568, 256), (256, 1), 0), reinterpret_tensor(arg59_1, (256, 128), (1, 256), 0), out=buf77)
    del arg59_1
    buf78 = buf72; del buf72  # reuse
    buf79 = reinterpret_tensor(buf71, (8, 196, 128), (25088, 128, 1), 0); del buf71  # reuse
    cpp_fused_add_clone_22(c_void_p(buf78.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf79.data_ptr()))
    del arg267_1
    del arg268_1
    del arg60_1
    del arg61_1
    buf80 = reinterpret_tensor(buf76, (1568, 256), (256, 1), 0); del buf76  # reuse
    # Source Nodes: [x_39], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf79, (1568, 128), (128, 1), 0), reinterpret_tensor(arg62_1, (128, 256), (1, 128), 0), out=buf80)
    del arg62_1
    buf81 = reinterpret_tensor(buf61, (8, 4, 196, 16), (12544, 3136, 16, 1), 0); del buf61  # reuse
    buf82 = reinterpret_tensor(buf60, (8, 4, 16, 196), (12544, 3136, 196, 1), 0); del buf60  # reuse
    cpp_fused_clone_23(c_void_p(buf80.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()))
    buf83 = reinterpret_tensor(buf67, (32, 196, 196), (38416, 196, 1), 0); del buf67  # reuse
    # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf81, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf82, (32, 16, 196), (3136, 196, 1), 0), out=buf83)
    buf84 = empty((4, 196, 196), device='cpu', dtype=torch.float32)
    buf85 = buf66; del buf66  # reuse
    buf86 = reinterpret_tensor(buf83, (8, 4, 196, 196), (153664, 38416, 196, 1), 0); del buf83  # reuse
    buf87 = buf64; del buf64  # reuse
    buf88 = buf86; del buf86  # reuse
    buf89 = reinterpret_tensor(buf79, (8, 4, 196, 32), (25088, 6272, 32, 1), 0); del buf79  # reuse
    cpp_fused__softmax_add_clone_index_mul_24(c_void_p(buf88.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()), c_void_p(buf87.data_ptr()), c_void_p(buf89.data_ptr()))
    del arg211_1
    del arg270_1
    del arg271_1
    del arg3_1
    del arg63_1
    del arg64_1
    del buf85
    del buf87
    buf90 = reinterpret_tensor(buf77, (32, 196, 32), (6272, 32, 1), 0); del buf77  # reuse
    # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf88, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf89, (32, 196, 32), (6272, 32, 1), 0), out=buf90)
    del buf88
    buf91 = reinterpret_tensor(buf89, (8, 196, 128), (25088, 128, 1), 0); del buf89  # reuse
    cpp_fused_hardswish_25(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    buf92 = reinterpret_tensor(buf90, (1568, 128), (128, 1), 0); del buf90  # reuse
    # Source Nodes: [x_41], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf91, (1568, 128), (128, 1), 0), reinterpret_tensor(arg65_1, (128, 128), (1, 128), 0), out=buf92)
    del arg65_1
    buf93 = buf78; del buf78  # reuse
    buf94 = buf91; del buf91  # reuse
    cpp_fused_add_clone_26(c_void_p(buf93.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg273_1
    del arg274_1
    del arg66_1
    del arg67_1
    buf95 = buf80; del buf80  # reuse
    # Source Nodes: [x_44], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf94, (1568, 128), (128, 1), 0), reinterpret_tensor(arg68_1, (128, 256), (1, 128), 0), out=buf95)
    del arg68_1
    buf96 = buf95; del buf95  # reuse
    buf97 = reinterpret_tensor(buf96, (8, 196, 256), (50176, 256, 1), 0); del buf96  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_27(c_void_p(buf97.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()))
    del arg276_1
    del arg277_1
    del arg69_1
    del arg70_1
    buf98 = reinterpret_tensor(buf94, (1568, 128), (128, 1), 0); del buf94  # reuse
    # Source Nodes: [x_48], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf97, (1568, 256), (256, 1), 0), reinterpret_tensor(arg71_1, (256, 128), (1, 256), 0), out=buf98)
    del arg71_1
    buf99 = buf93; del buf93  # reuse
    buf100 = reinterpret_tensor(buf92, (8, 196, 128), (25088, 128, 1), 0); del buf92  # reuse
    cpp_fused_add_clone_28(c_void_p(buf99.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf100.data_ptr()))
    del arg279_1
    del arg280_1
    del arg72_1
    del arg73_1
    del buf98
    buf101 = empty((1568, 640), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_52], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf100, (1568, 128), (128, 1), 0), reinterpret_tensor(arg74_1, (128, 640), (1, 128), 0), out=buf101)
    del arg74_1
    buf102 = empty((8, 7, 7, 128), device='cpu', dtype=torch.float32)
    cpp_fused_clone_29(c_void_p(buf99.data_ptr()), c_void_p(buf102.data_ptr()))
    buf103 = empty((392, 128), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_55], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf102, (392, 128), (128, 1), 0), reinterpret_tensor(arg77_1, (128, 128), (1, 128), 0), out=buf103)
    del arg77_1
    buf104 = reinterpret_tensor(buf102, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf102  # reuse
    buf105 = reinterpret_tensor(buf99, (8, 8, 16, 196), (25088, 3136, 196, 1), 0); del buf99  # reuse
    cpp_fused_clone_30(c_void_p(buf103.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg285_1
    del arg286_1
    del arg78_1
    del arg79_1
    buf106 = empty((64, 49, 196), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf104, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf105, (64, 16, 196), (3136, 196, 1), 0), out=buf106)
    buf107 = empty((8, 49, 196), device='cpu', dtype=torch.float32)
    buf108 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cpu', dtype=torch.float32)
    buf109 = reinterpret_tensor(buf106, (8, 8, 49, 196), (76832, 9604, 196, 1), 0); del buf106  # reuse
    buf110 = empty_strided((8, 8, 49, 1), (392, 49, 1, 3136), device='cpu', dtype=torch.float32)
    buf111 = buf109; del buf109  # reuse
    buf112 = reinterpret_tensor(buf8, (8, 8, 196, 64), (100352, 12544, 64, 1), 0); del buf8  # reuse
    cpp_fused__softmax_add_clone_index_mul_31(c_void_p(buf111.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()))
    del arg212_1
    del arg282_1
    del arg283_1
    del arg4_1
    del arg75_1
    del arg76_1
    del buf101
    buf113 = reinterpret_tensor(buf105, (64, 49, 64), (3136, 64, 1), 0); del buf105  # reuse
    # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf111, (64, 49, 196), (9604, 196, 1), 0), reinterpret_tensor(buf112, (64, 196, 64), (12544, 64, 1), 0), out=buf113)
    del buf111
    del buf112
    buf114 = reinterpret_tensor(buf100, (8, 49, 512), (25088, 512, 1), 0); del buf100  # reuse
    cpp_fused_hardswish_32(c_void_p(buf113.data_ptr()), c_void_p(buf114.data_ptr()))
    del buf113
    buf115 = reinterpret_tensor(buf82, (392, 256), (256, 1), 0); del buf82  # reuse
    # Source Nodes: [x_57], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf114, (392, 512), (512, 1), 0), reinterpret_tensor(arg80_1, (512, 256), (1, 512), 0), out=buf115)
    del arg80_1
    buf116 = buf115; del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_33(c_void_p(buf116.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()))
    del arg288_1
    del arg289_1
    del arg81_1
    del arg82_1
    buf117 = reinterpret_tensor(buf114, (392, 512), (512, 1), 0); del buf114  # reuse
    # Source Nodes: [x_60], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf116, (392, 256), (256, 1), 0), reinterpret_tensor(arg83_1, (256, 512), (1, 256), 0), out=buf117)
    del arg83_1
    buf118 = buf117; del buf117  # reuse
    buf119 = reinterpret_tensor(buf118, (8, 49, 512), (25088, 512, 1), 0); del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_34(c_void_p(buf119.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg291_1
    del arg292_1
    del arg84_1
    del arg85_1
    buf120 = reinterpret_tensor(buf81, (392, 256), (256, 1), 0); del buf81  # reuse
    # Source Nodes: [x_64], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf119, (392, 512), (512, 1), 0), reinterpret_tensor(arg86_1, (512, 256), (1, 512), 0), out=buf120)
    del arg86_1
    buf121 = reinterpret_tensor(buf116, (8, 49, 256), (12544, 256, 1), 0); del buf116  # reuse
    cpp_fused_add_35(c_void_p(buf121.data_ptr()), c_void_p(buf120.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg88_1.data_ptr()))
    del arg294_1
    del arg295_1
    del arg87_1
    del arg88_1
    buf122 = reinterpret_tensor(buf119, (392, 512), (512, 1), 0); del buf119  # reuse
    # Source Nodes: [x_68], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf121, (392, 256), (256, 1), 0), reinterpret_tensor(arg89_1, (256, 512), (1, 256), 0), out=buf122)
    del arg89_1
    buf123 = buf104; del buf104  # reuse
    buf124 = reinterpret_tensor(buf103, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf103  # reuse
    cpp_fused_clone_36(c_void_p(buf122.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()))
    buf125 = empty((64, 49, 49), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf123, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf124, (64, 16, 49), (784, 49, 1), 0), out=buf125)
    buf126 = empty((8, 49, 49), device='cpu', dtype=torch.float32)
    buf127 = buf110; del buf110  # reuse
    buf128 = reinterpret_tensor(buf125, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf125  # reuse
    buf129 = buf108; del buf108  # reuse
    buf130 = buf128; del buf128  # reuse
    buf131 = reinterpret_tensor(buf120, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf120  # reuse
    cpp_fused__softmax_add_clone_index_mul_37(c_void_p(buf130.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg213_1
    del arg297_1
    del arg298_1
    del arg5_1
    del arg90_1
    del arg91_1
    buf132 = empty((64, 49, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf130, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf131, (64, 49, 32), (1568, 32, 1), 0), out=buf132)
    buf133 = reinterpret_tensor(buf131, (8, 49, 256), (12544, 256, 1), 0); del buf131  # reuse
    cpp_fused_hardswish_38(c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    buf134 = reinterpret_tensor(buf132, (392, 256), (256, 1), 0); del buf132  # reuse
    # Source Nodes: [x_70], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf133, (392, 256), (256, 1), 0), reinterpret_tensor(arg92_1, (256, 256), (1, 256), 0), out=buf134)
    del arg92_1
    buf135 = buf121; del buf121  # reuse
    cpp_fused_add_39(c_void_p(buf135.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()))
    del arg300_1
    del arg301_1
    del arg93_1
    del arg94_1
    buf136 = buf122; del buf122  # reuse
    # Source Nodes: [x_73], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf135, (392, 256), (256, 1), 0), reinterpret_tensor(arg95_1, (256, 512), (1, 256), 0), out=buf136)
    del arg95_1
    buf137 = buf136; del buf136  # reuse
    buf138 = reinterpret_tensor(buf137, (8, 49, 512), (25088, 512, 1), 0); del buf137  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_40(c_void_p(buf138.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg303_1
    del arg304_1
    del arg96_1
    del arg97_1
    buf139 = buf134; del buf134  # reuse
    # Source Nodes: [x_77], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf138, (392, 512), (512, 1), 0), reinterpret_tensor(arg98_1, (512, 256), (1, 512), 0), out=buf139)
    del arg98_1
    buf140 = buf135; del buf135  # reuse
    cpp_fused_add_41(c_void_p(buf140.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()))
    del arg100_1
    del arg306_1
    del arg307_1
    del arg99_1
    buf141 = reinterpret_tensor(buf138, (392, 512), (512, 1), 0); del buf138  # reuse
    # Source Nodes: [x_80], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf140, (392, 256), (256, 1), 0), reinterpret_tensor(arg101_1, (256, 512), (1, 256), 0), out=buf141)
    del arg101_1
    buf142 = reinterpret_tensor(buf124, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf124  # reuse
    buf143 = reinterpret_tensor(buf123, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf123  # reuse
    cpp_fused_clone_42(c_void_p(buf141.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    buf144 = reinterpret_tensor(buf130, (64, 49, 49), (2401, 49, 1), 0); del buf130  # reuse
    # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf142, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf143, (64, 16, 49), (784, 49, 1), 0), out=buf144)
    buf145 = empty((8, 49, 49), device='cpu', dtype=torch.float32)
    buf146 = buf129; del buf129  # reuse
    buf147 = reinterpret_tensor(buf144, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf144  # reuse
    buf148 = buf127; del buf127  # reuse
    buf149 = buf147; del buf147  # reuse
    buf150 = reinterpret_tensor(buf139, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf139  # reuse
    cpp_fused__softmax_add_clone_index_mul_43(c_void_p(buf149.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf150.data_ptr()))
    del arg102_1
    del arg103_1
    del arg214_1
    del arg309_1
    del arg310_1
    del arg6_1
    buf151 = reinterpret_tensor(buf133, (64, 49, 32), (1568, 32, 1), 0); del buf133  # reuse
    # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf149, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf150, (64, 49, 32), (1568, 32, 1), 0), out=buf151)
    buf152 = reinterpret_tensor(buf150, (8, 49, 256), (12544, 256, 1), 0); del buf150  # reuse
    cpp_fused_hardswish_44(c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()))
    buf153 = reinterpret_tensor(buf151, (392, 256), (256, 1), 0); del buf151  # reuse
    # Source Nodes: [x_82], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf152, (392, 256), (256, 1), 0), reinterpret_tensor(arg104_1, (256, 256), (1, 256), 0), out=buf153)
    del arg104_1
    buf154 = buf140; del buf140  # reuse
    cpp_fused_add_45(c_void_p(buf154.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg106_1.data_ptr()))
    del arg105_1
    del arg106_1
    del arg312_1
    del arg313_1
    buf155 = buf141; del buf141  # reuse
    # Source Nodes: [x_85], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf154, (392, 256), (256, 1), 0), reinterpret_tensor(arg107_1, (256, 512), (1, 256), 0), out=buf155)
    del arg107_1
    buf156 = buf155; del buf155  # reuse
    buf157 = reinterpret_tensor(buf156, (8, 49, 512), (25088, 512, 1), 0); del buf156  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_46(c_void_p(buf157.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg315_1
    del arg316_1
    buf158 = buf153; del buf153  # reuse
    # Source Nodes: [x_89], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf157, (392, 512), (512, 1), 0), reinterpret_tensor(arg110_1, (512, 256), (1, 512), 0), out=buf158)
    del arg110_1
    buf159 = buf154; del buf154  # reuse
    cpp_fused_add_47(c_void_p(buf159.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()))
    del arg111_1
    del arg112_1
    del arg318_1
    del arg319_1
    buf160 = reinterpret_tensor(buf157, (392, 512), (512, 1), 0); del buf157  # reuse
    # Source Nodes: [x_92], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf159, (392, 256), (256, 1), 0), reinterpret_tensor(arg113_1, (256, 512), (1, 256), 0), out=buf160)
    del arg113_1
    buf161 = reinterpret_tensor(buf143, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf143  # reuse
    buf162 = reinterpret_tensor(buf142, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf142  # reuse
    cpp_fused_clone_48(c_void_p(buf160.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf162.data_ptr()))
    buf163 = reinterpret_tensor(buf149, (64, 49, 49), (2401, 49, 1), 0); del buf149  # reuse
    # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf161, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf162, (64, 16, 49), (784, 49, 1), 0), out=buf163)
    buf164 = empty((8, 49, 49), device='cpu', dtype=torch.float32)
    buf165 = buf148; del buf148  # reuse
    buf166 = reinterpret_tensor(buf163, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf163  # reuse
    buf167 = buf146; del buf146  # reuse
    buf168 = buf166; del buf166  # reuse
    buf169 = reinterpret_tensor(buf158, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf158  # reuse
    cpp_fused__softmax_add_clone_index_mul_49(c_void_p(buf168.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf169.data_ptr()))
    del arg114_1
    del arg115_1
    del arg215_1
    del arg321_1
    del arg322_1
    del arg7_1
    buf170 = reinterpret_tensor(buf152, (64, 49, 32), (1568, 32, 1), 0); del buf152  # reuse
    # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf168, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf169, (64, 49, 32), (1568, 32, 1), 0), out=buf170)
    buf171 = reinterpret_tensor(buf169, (8, 49, 256), (12544, 256, 1), 0); del buf169  # reuse
    cpp_fused_hardswish_50(c_void_p(buf170.data_ptr()), c_void_p(buf171.data_ptr()))
    buf172 = reinterpret_tensor(buf170, (392, 256), (256, 1), 0); del buf170  # reuse
    # Source Nodes: [x_94], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf171, (392, 256), (256, 1), 0), reinterpret_tensor(arg116_1, (256, 256), (1, 256), 0), out=buf172)
    del arg116_1
    buf173 = buf159; del buf159  # reuse
    cpp_fused_add_51(c_void_p(buf173.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()))
    del arg117_1
    del arg118_1
    del arg324_1
    del arg325_1
    buf174 = buf160; del buf160  # reuse
    # Source Nodes: [x_97], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf173, (392, 256), (256, 1), 0), reinterpret_tensor(arg119_1, (256, 512), (1, 256), 0), out=buf174)
    del arg119_1
    buf175 = buf174; del buf174  # reuse
    buf176 = reinterpret_tensor(buf175, (8, 49, 512), (25088, 512, 1), 0); del buf175  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_52(c_void_p(buf176.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg120_1
    del arg121_1
    del arg327_1
    del arg328_1
    buf177 = buf172; del buf172  # reuse
    # Source Nodes: [x_101], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf176, (392, 512), (512, 1), 0), reinterpret_tensor(arg122_1, (512, 256), (1, 512), 0), out=buf177)
    del arg122_1
    buf178 = buf173; del buf173  # reuse
    cpp_fused_add_53(c_void_p(buf178.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()))
    del arg123_1
    del arg124_1
    del arg330_1
    del arg331_1
    buf179 = reinterpret_tensor(buf176, (392, 512), (512, 1), 0); del buf176  # reuse
    # Source Nodes: [x_104], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf178, (392, 256), (256, 1), 0), reinterpret_tensor(arg125_1, (256, 512), (1, 256), 0), out=buf179)
    del arg125_1
    buf180 = reinterpret_tensor(buf162, (8, 8, 49, 16), (6272, 784, 16, 1), 0); del buf162  # reuse
    buf181 = reinterpret_tensor(buf161, (8, 8, 16, 49), (6272, 784, 49, 1), 0); del buf161  # reuse
    cpp_fused_clone_54(c_void_p(buf179.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    buf182 = reinterpret_tensor(buf168, (64, 49, 49), (2401, 49, 1), 0); del buf168  # reuse
    # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf180, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf181, (64, 16, 49), (784, 49, 1), 0), out=buf182)
    del buf180
    del buf181
    buf183 = empty((8, 49, 49), device='cpu', dtype=torch.float32)
    buf184 = buf167; del buf167  # reuse
    buf185 = reinterpret_tensor(buf182, (8, 8, 49, 49), (19208, 2401, 49, 1), 0); del buf182  # reuse
    buf186 = buf165; del buf165  # reuse
    buf187 = buf185; del buf185  # reuse
    buf188 = reinterpret_tensor(buf177, (8, 8, 49, 32), (12544, 1568, 32, 1), 0); del buf177  # reuse
    cpp_fused__softmax_add_clone_index_mul_55(c_void_p(buf187.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf188.data_ptr()))
    del arg126_1
    del arg127_1
    del arg216_1
    del arg333_1
    del arg334_1
    del arg8_1
    del buf184
    del buf186
    buf189 = reinterpret_tensor(buf171, (64, 49, 32), (1568, 32, 1), 0); del buf171  # reuse
    # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf187, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf188, (64, 49, 32), (1568, 32, 1), 0), out=buf189)
    del buf187
    buf190 = reinterpret_tensor(buf188, (8, 49, 256), (12544, 256, 1), 0); del buf188  # reuse
    cpp_fused_hardswish_56(c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    buf191 = reinterpret_tensor(buf189, (392, 256), (256, 1), 0); del buf189  # reuse
    # Source Nodes: [x_106], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf190, (392, 256), (256, 1), 0), reinterpret_tensor(arg128_1, (256, 256), (1, 256), 0), out=buf191)
    del arg128_1
    del buf190
    buf192 = buf178; del buf178  # reuse
    cpp_fused_add_57(c_void_p(buf192.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg336_1
    del arg337_1
    buf193 = buf179; del buf179  # reuse
    # Source Nodes: [x_109], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf192, (392, 256), (256, 1), 0), reinterpret_tensor(arg131_1, (256, 512), (1, 256), 0), out=buf193)
    del arg131_1
    buf194 = buf193; del buf193  # reuse
    buf195 = reinterpret_tensor(buf194, (8, 49, 512), (25088, 512, 1), 0); del buf194  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_58(c_void_p(buf195.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()))
    del arg132_1
    del arg133_1
    del arg339_1
    del arg340_1
    buf196 = buf191; del buf191  # reuse
    # Source Nodes: [x_113], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf195, (392, 512), (512, 1), 0), reinterpret_tensor(arg134_1, (512, 256), (1, 512), 0), out=buf196)
    del arg134_1
    del buf195
    buf197 = buf192; del buf192  # reuse
    cpp_fused_add_59(c_void_p(buf197.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg136_1.data_ptr()))
    del arg135_1
    del arg136_1
    del arg342_1
    del arg343_1
    buf198 = empty((392, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_117], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf197, (392, 256), (256, 1), 0), reinterpret_tensor(arg137_1, (256, 1280), (1, 256), 0), out=buf198)
    del arg137_1
    buf199 = empty((8, 4, 4, 256), device='cpu', dtype=torch.float32)
    cpp_fused_clone_60(c_void_p(buf197.data_ptr()), c_void_p(buf199.data_ptr()))
    buf200 = empty((128, 256), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_120], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf199, (128, 256), (256, 1), 0), reinterpret_tensor(arg140_1, (256, 256), (1, 256), 0), out=buf200)
    del arg140_1
    buf201 = reinterpret_tensor(buf199, (8, 16, 16, 16), (4096, 256, 16, 1), 0); del buf199  # reuse
    buf202 = reinterpret_tensor(buf197, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf197  # reuse
    cpp_fused_clone_61(c_void_p(buf200.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(buf202.data_ptr()))
    del arg141_1
    del arg142_1
    del arg348_1
    del arg349_1
    del buf200
    buf203 = reinterpret_tensor(buf196, (128, 16, 49), (784, 49, 1), 0); del buf196  # reuse
    # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf201, (128, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf202, (128, 16, 49), (784, 49, 1), 0), out=buf203)
    del buf201
    del buf202
    buf204 = empty((16, 16, 49), device='cpu', dtype=torch.float32)
    buf205 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cpu', dtype=torch.float32)
    buf206 = reinterpret_tensor(buf203, (8, 16, 16, 49), (12544, 784, 49, 1), 0); del buf203  # reuse
    buf207 = empty_strided((8, 16, 16, 1), (256, 16, 1, 2048), device='cpu', dtype=torch.float32)
    buf208 = buf206; del buf206  # reuse
    buf209 = reinterpret_tensor(buf97, (8, 16, 49, 64), (50176, 3136, 64, 1), 0); del buf97  # reuse
    cpp_fused__softmax_add_clone_index_mul_62(c_void_p(buf208.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf198.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(buf209.data_ptr()))
    del arg138_1
    del arg139_1
    del arg217_1
    del arg345_1
    del arg346_1
    del arg9_1
    del buf198
    del buf205
    del buf207
    buf210 = empty((128, 16, 64), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf208, (128, 16, 49), (784, 49, 1), 0), reinterpret_tensor(buf209, (128, 49, 64), (3136, 64, 1), 0), out=buf210)
    del buf208
    del buf209
    buf211 = empty((8, 16, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_hardswish_63(c_void_p(buf210.data_ptr()), c_void_p(buf211.data_ptr()))
    del buf210
    buf212 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_122], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf211, (128, 1024), (1024, 1), 0), reinterpret_tensor(arg143_1, (1024, 384), (1, 1024), 0), out=buf212)
    del arg143_1
    del buf211
    buf213 = buf212; del buf212  # reuse
    cpp_fused__native_batch_norm_legit_no_training_64(c_void_p(buf213.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()))
    del arg144_1
    del arg145_1
    del arg351_1
    del arg352_1
    buf214 = empty((128, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_125], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf213, (128, 384), (384, 1), 0), reinterpret_tensor(arg146_1, (384, 768), (1, 384), 0), out=buf214)
    del arg146_1
    buf215 = buf214; del buf214  # reuse
    buf216 = reinterpret_tensor(buf215, (8, 16, 768), (12288, 768, 1), 0); del buf215  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_65(c_void_p(buf216.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg148_1.data_ptr()))
    del arg147_1
    del arg148_1
    del arg354_1
    del arg355_1
    buf217 = empty((128, 384), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_129], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf216, (128, 768), (768, 1), 0), reinterpret_tensor(arg149_1, (768, 384), (1, 768), 0), out=buf217)
    del arg149_1
    buf218 = reinterpret_tensor(buf213, (8, 16, 384), (6144, 384, 1), 0); del buf213  # reuse
    cpp_fused_add_66(c_void_p(buf218.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()))
    del arg150_1
    del arg151_1
    del arg357_1
    del arg358_1
    buf219 = reinterpret_tensor(buf216, (128, 768), (768, 1), 0); del buf216  # reuse
    # Source Nodes: [x_133], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf218, (128, 384), (384, 1), 0), reinterpret_tensor(arg152_1, (384, 768), (1, 384), 0), out=buf219)
    del arg152_1
    buf220 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    buf221 = empty((8, 12, 16, 16), device='cpu', dtype=torch.float32)
    cpp_fused_clone_67(c_void_p(buf219.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()))
    buf222 = empty((96, 16, 16), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf220, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf221, (96, 16, 16), (256, 16, 1), 0), out=buf222)
    buf223 = empty((12, 16, 16), device='cpu', dtype=torch.float32)
    buf224 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cpu', dtype=torch.float32)
    buf225 = reinterpret_tensor(buf222, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf222  # reuse
    buf226 = empty_strided((8, 12, 16, 1), (192, 16, 1, 1536), device='cpu', dtype=torch.float32)
    buf227 = buf225; del buf225  # reuse
    buf228 = reinterpret_tensor(buf217, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf217  # reuse
    cpp_fused__softmax_add_clone_index_mul_68(c_void_p(buf227.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg10_1
    del arg153_1
    del arg154_1
    del arg218_1
    del arg360_1
    del arg361_1
    buf229 = empty((96, 16, 32), device='cpu', dtype=torch.float32)
    # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf227, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf228, (96, 16, 32), (512, 32, 1), 0), out=buf229)
    buf230 = reinterpret_tensor(buf228, (8, 16, 384), (6144, 384, 1), 0); del buf228  # reuse
    cpp_fused_hardswish_69(c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    buf231 = reinterpret_tensor(buf229, (128, 384), (384, 1), 0); del buf229  # reuse
    # Source Nodes: [x_135], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf230, (128, 384), (384, 1), 0), reinterpret_tensor(arg155_1, (384, 384), (1, 384), 0), out=buf231)
    del arg155_1
    buf232 = buf218; del buf218  # reuse
    cpp_fused_add_70(c_void_p(buf232.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()))
    del arg156_1
    del arg157_1
    del arg363_1
    del arg364_1
    buf233 = buf219; del buf219  # reuse
    # Source Nodes: [x_138], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf232, (128, 384), (384, 1), 0), reinterpret_tensor(arg158_1, (384, 768), (1, 384), 0), out=buf233)
    del arg158_1
    buf234 = buf233; del buf233  # reuse
    buf235 = reinterpret_tensor(buf234, (8, 16, 768), (12288, 768, 1), 0); del buf234  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_71(c_void_p(buf235.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(arg160_1.data_ptr()))
    del arg159_1
    del arg160_1
    del arg366_1
    del arg367_1
    buf236 = buf231; del buf231  # reuse
    # Source Nodes: [x_142], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf235, (128, 768), (768, 1), 0), reinterpret_tensor(arg161_1, (768, 384), (1, 768), 0), out=buf236)
    del arg161_1
    buf237 = buf232; del buf232  # reuse
    cpp_fused_add_72(c_void_p(buf237.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()))
    del arg162_1
    del arg163_1
    del arg369_1
    del arg370_1
    buf238 = reinterpret_tensor(buf235, (128, 768), (768, 1), 0); del buf235  # reuse
    # Source Nodes: [x_145], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf237, (128, 384), (384, 1), 0), reinterpret_tensor(arg164_1, (384, 768), (1, 384), 0), out=buf238)
    del arg164_1
    buf239 = buf227; del buf227  # reuse
    buf240 = buf221; del buf221  # reuse
    cpp_fused_clone_73(c_void_p(buf238.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    buf241 = reinterpret_tensor(buf220, (96, 16, 16), (256, 16, 1), 0); del buf220  # reuse
    # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf239, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf240, (96, 16, 16), (256, 16, 1), 0), out=buf241)
    buf242 = empty((12, 16, 16), device='cpu', dtype=torch.float32)
    buf243 = buf226; del buf226  # reuse
    buf244 = reinterpret_tensor(buf241, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf241  # reuse
    buf245 = buf224; del buf224  # reuse
    buf246 = buf244; del buf244  # reuse
    buf247 = reinterpret_tensor(buf236, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf236  # reuse
    cpp_fused__softmax_add_clone_index_mul_74(c_void_p(buf246.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg11_1
    del arg165_1
    del arg166_1
    del arg219_1
    del arg372_1
    del arg373_1
    buf248 = reinterpret_tensor(buf230, (96, 16, 32), (512, 32, 1), 0); del buf230  # reuse
    # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf246, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf247, (96, 16, 32), (512, 32, 1), 0), out=buf248)
    buf249 = reinterpret_tensor(buf247, (8, 16, 384), (6144, 384, 1), 0); del buf247  # reuse
    cpp_fused_hardswish_75(c_void_p(buf248.data_ptr()), c_void_p(buf249.data_ptr()))
    buf250 = reinterpret_tensor(buf248, (128, 384), (384, 1), 0); del buf248  # reuse
    # Source Nodes: [x_147], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf249, (128, 384), (384, 1), 0), reinterpret_tensor(arg167_1, (384, 384), (1, 384), 0), out=buf250)
    del arg167_1
    buf251 = buf237; del buf237  # reuse
    cpp_fused_add_76(c_void_p(buf251.data_ptr()), c_void_p(buf250.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()))
    del arg168_1
    del arg169_1
    del arg375_1
    del arg376_1
    buf252 = buf238; del buf238  # reuse
    # Source Nodes: [x_150], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf251, (128, 384), (384, 1), 0), reinterpret_tensor(arg170_1, (384, 768), (1, 384), 0), out=buf252)
    del arg170_1
    buf253 = buf252; del buf252  # reuse
    buf254 = reinterpret_tensor(buf253, (8, 16, 768), (12288, 768, 1), 0); del buf253  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_77(c_void_p(buf254.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg172_1.data_ptr()))
    del arg171_1
    del arg172_1
    del arg378_1
    del arg379_1
    buf255 = buf250; del buf250  # reuse
    # Source Nodes: [x_154], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf254, (128, 768), (768, 1), 0), reinterpret_tensor(arg173_1, (768, 384), (1, 768), 0), out=buf255)
    del arg173_1
    buf256 = buf251; del buf251  # reuse
    cpp_fused_add_78(c_void_p(buf256.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()))
    del arg174_1
    del arg175_1
    del arg381_1
    del arg382_1
    buf257 = reinterpret_tensor(buf254, (128, 768), (768, 1), 0); del buf254  # reuse
    # Source Nodes: [x_157], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf256, (128, 384), (384, 1), 0), reinterpret_tensor(arg176_1, (384, 768), (1, 384), 0), out=buf257)
    del arg176_1
    buf258 = buf246; del buf246  # reuse
    buf259 = buf240; del buf240  # reuse
    cpp_fused_clone_79(c_void_p(buf257.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf259.data_ptr()))
    buf260 = reinterpret_tensor(buf239, (96, 16, 16), (256, 16, 1), 0); del buf239  # reuse
    # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf258, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf259, (96, 16, 16), (256, 16, 1), 0), out=buf260)
    buf261 = empty((12, 16, 16), device='cpu', dtype=torch.float32)
    buf262 = buf245; del buf245  # reuse
    buf263 = reinterpret_tensor(buf260, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf260  # reuse
    buf264 = buf243; del buf243  # reuse
    buf265 = buf263; del buf263  # reuse
    buf266 = reinterpret_tensor(buf255, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf255  # reuse
    cpp_fused__softmax_add_clone_index_mul_80(c_void_p(buf265.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()))
    del arg12_1
    del arg177_1
    del arg178_1
    del arg220_1
    del arg384_1
    del arg385_1
    buf267 = reinterpret_tensor(buf249, (96, 16, 32), (512, 32, 1), 0); del buf249  # reuse
    # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf265, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf266, (96, 16, 32), (512, 32, 1), 0), out=buf267)
    buf268 = reinterpret_tensor(buf266, (8, 16, 384), (6144, 384, 1), 0); del buf266  # reuse
    cpp_fused_hardswish_81(c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()))
    buf269 = reinterpret_tensor(buf267, (128, 384), (384, 1), 0); del buf267  # reuse
    # Source Nodes: [x_159], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf268, (128, 384), (384, 1), 0), reinterpret_tensor(arg179_1, (384, 384), (1, 384), 0), out=buf269)
    del arg179_1
    buf270 = buf256; del buf256  # reuse
    cpp_fused_add_82(c_void_p(buf270.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()))
    del arg180_1
    del arg181_1
    del arg387_1
    del arg388_1
    buf271 = buf257; del buf257  # reuse
    # Source Nodes: [x_162], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf270, (128, 384), (384, 1), 0), reinterpret_tensor(arg182_1, (384, 768), (1, 384), 0), out=buf271)
    del arg182_1
    buf272 = buf271; del buf271  # reuse
    buf273 = reinterpret_tensor(buf272, (8, 16, 768), (12288, 768, 1), 0); del buf272  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_83(c_void_p(buf273.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()))
    del arg183_1
    del arg184_1
    del arg390_1
    del arg391_1
    buf274 = buf269; del buf269  # reuse
    # Source Nodes: [x_166], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf273, (128, 768), (768, 1), 0), reinterpret_tensor(arg185_1, (768, 384), (1, 768), 0), out=buf274)
    del arg185_1
    buf275 = buf270; del buf270  # reuse
    cpp_fused_add_84(c_void_p(buf275.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()))
    del arg186_1
    del arg187_1
    del arg393_1
    del arg394_1
    buf276 = reinterpret_tensor(buf273, (128, 768), (768, 1), 0); del buf273  # reuse
    # Source Nodes: [x_169], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf275, (128, 384), (384, 1), 0), reinterpret_tensor(arg188_1, (384, 768), (1, 384), 0), out=buf276)
    del arg188_1
    buf277 = buf265; del buf265  # reuse
    buf278 = buf259; del buf259  # reuse
    cpp_fused_clone_85(c_void_p(buf276.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    buf279 = reinterpret_tensor(buf258, (96, 16, 16), (256, 16, 1), 0); del buf258  # reuse
    # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf277, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf278, (96, 16, 16), (256, 16, 1), 0), out=buf279)
    del buf277
    del buf278
    buf280 = empty((12, 16, 16), device='cpu', dtype=torch.float32)
    buf281 = buf264; del buf264  # reuse
    buf282 = reinterpret_tensor(buf279, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf279  # reuse
    buf283 = buf262; del buf262  # reuse
    buf284 = buf282; del buf282  # reuse
    buf285 = reinterpret_tensor(buf274, (8, 12, 16, 32), (6144, 512, 32, 1), 0); del buf274  # reuse
    cpp_fused__softmax_add_clone_index_mul_86(c_void_p(buf284.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf285.data_ptr()))
    del arg13_1
    del arg189_1
    del arg190_1
    del arg221_1
    del arg396_1
    del arg397_1
    del buf281
    del buf283
    buf286 = reinterpret_tensor(buf268, (96, 16, 32), (512, 32, 1), 0); del buf268  # reuse
    # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf284, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf285, (96, 16, 32), (512, 32, 1), 0), out=buf286)
    del buf284
    buf287 = reinterpret_tensor(buf285, (8, 16, 384), (6144, 384, 1), 0); del buf285  # reuse
    cpp_fused_hardswish_87(c_void_p(buf286.data_ptr()), c_void_p(buf287.data_ptr()))
    buf288 = reinterpret_tensor(buf286, (128, 384), (384, 1), 0); del buf286  # reuse
    # Source Nodes: [x_171], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf287, (128, 384), (384, 1), 0), reinterpret_tensor(arg191_1, (384, 384), (1, 384), 0), out=buf288)
    del arg191_1
    del buf287
    buf289 = buf275; del buf275  # reuse
    cpp_fused_add_88(c_void_p(buf289.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg399_1
    del arg400_1
    buf290 = buf276; del buf276  # reuse
    # Source Nodes: [x_174], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf289, (128, 384), (384, 1), 0), reinterpret_tensor(arg194_1, (384, 768), (1, 384), 0), out=buf290)
    del arg194_1
    buf291 = buf290; del buf290  # reuse
    buf292 = reinterpret_tensor(buf291, (8, 16, 768), (12288, 768, 1), 0); del buf291  # reuse
    cpp_fused__native_batch_norm_legit_no_training_hardswish_89(c_void_p(buf292.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()))
    del arg195_1
    del arg196_1
    del arg402_1
    del arg403_1
    buf293 = buf288; del buf288  # reuse
    # Source Nodes: [x_178], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf292, (128, 768), (768, 1), 0), reinterpret_tensor(arg197_1, (768, 384), (1, 768), 0), out=buf293)
    del arg197_1
    del buf292
    buf294 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf295 = empty((8, 384), device='cpu', dtype=torch.float32)
    buf297 = empty((8, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_mean_90(c_void_p(buf289.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(buf297.data_ptr()))
    del arg198_1
    del arg199_1
    del arg200_1
    del arg201_1
    del arg204_1
    del arg205_1
    del arg405_1
    del arg406_1
    del arg408_1
    del arg409_1
    del arg411_1
    del arg412_1
    del buf289
    del buf293
    del buf294
    buf296 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_bn, x_183, x_184, x_185], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.addmm, aten.mean]
    extern_kernels.addmm(arg203_1, buf295, reinterpret_tensor(arg202_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf296)
    del arg202_1
    del arg203_1
    del buf295
    buf298 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [l__mod___head_dist_bn, x_183, x_184, x_dist], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.addmm, aten.mean]
    extern_kernels.addmm(arg207_1, buf297, reinterpret_tensor(arg206_1, (384, 1000), (1, 384), 0), alpha=1, beta=1, out=buf298)
    del arg206_1
    del arg207_1
    del buf297
    buf299 = buf296; del buf296  # reuse
    cpp_fused_add_div_91(c_void_p(buf299.data_ptr()), c_void_p(buf298.data_ptr()))
    return (buf299, buf21, buf42, buf63, buf84, buf107, buf126, buf145, buf164, buf183, buf204, buf223, buf242, buf261, buf280, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((4, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((8, 196), (196, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((8, 49), (49, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((16, 49), (49, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((12, 16), (16, 1), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((256, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((640, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((128, 128), (128, 1), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((512, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((256, 512), (512, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((1280, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((256, 256), (256, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((384, 1024), (1024, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((384, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((768, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((384, 768), (768, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((1000, 384), (384, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    arg209_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    arg210_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    arg211_1 = rand_strided((196, 196), (196, 1), device='cpu', dtype=torch.int64)
    arg212_1 = rand_strided((49, 196), (196, 1), device='cpu', dtype=torch.int64)
    arg213_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg214_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg215_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg216_1 = rand_strided((49, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg217_1 = rand_strided((16, 49), (49, 1), device='cpu', dtype=torch.int64)
    arg218_1 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    arg219_1 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    arg220_1 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    arg221_1 = rand_strided((16, 16), (16, 1), device='cpu', dtype=torch.int64)
    arg222_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg225_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg228_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg231_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg234_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg237_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg240_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg243_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg246_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg249_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg252_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg255_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg258_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg261_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg264_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg267_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg270_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg273_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg276_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg279_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg282_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg285_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg288_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg291_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg294_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg297_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg300_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg303_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg306_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg309_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg312_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg315_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg318_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg321_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg324_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg327_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg330_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg333_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg336_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg339_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg342_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg345_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((1280, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg348_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg351_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg354_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg357_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg360_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg363_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg366_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg369_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg372_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg375_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg378_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg381_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg384_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg387_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg390_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg393_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg396_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg399_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg402_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg405_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg408_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg411_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg414_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
