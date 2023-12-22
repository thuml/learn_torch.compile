
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(89401L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (89401L*x1) + (268203L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (268203L*x0))] = tmp0;
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(177608L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(172872L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(172872L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(73L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(73L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(64L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(128L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(9408L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(9472L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(9536L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(18816L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(18880L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(18944L + x3 + (128L*x2) + (18816L*x1) + (1382976L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (4672L*x1) + (341056L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(42632L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (80L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (720L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (80L*x2) + (720L*x0)), static_cast<long>(80L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (720L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (80L*x2) + (720L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40328L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(35L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(192L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(192L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(384L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(13632L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(13824L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(14016L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(27264L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(27456L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(27648L + x3 + (384L*x2) + (27264L*x1) + (967872L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            tmp16.store(out_ptr0 + static_cast<long>(x3 + (192L*x2) + (6720L*x1) + (235200L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)), static_cast<long>(48L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(25L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_9 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(35L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x2);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(35);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x3);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-6912L) + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-6720L) + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x3);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-6528L) + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-192L) + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(192L + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x2);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(6528L + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(6720L + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(6912L + x1 + (192L*x3) + (6720L*x2) + (235200L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(36);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x2);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(35);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x3);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x2);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(35);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x3);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x2);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(35);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x3);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x2);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(35);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x3);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x2);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(35);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x3);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x2);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(35);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x3);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x2);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(35);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x3);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x2);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(35);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x3);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x2);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(35);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x3);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x1 + (192L*x3) + (6720L*x2) + (235200L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_10 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-64L) + x1 + (64L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-64L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-64L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-64L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-64L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(224);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-128L) + x1 + (96L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-128L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-128L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-128L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-128L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(256);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-224L) + x1 + (32L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-224L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-224L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-224L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-224L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp90;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)), static_cast<long>(48L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(25L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(35L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(35);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-9216L) + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-8960L) + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x2);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-8704L) + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-256L) + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(256L + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x1);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(8704L + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(8960L + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(9216L + x3 + (256L*x2) + (8960L*x1) + (313600L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(36);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x1);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(35);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x2);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x1);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(35);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x1);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(35);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x2);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(35);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x2);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(35);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(35);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x2);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x1);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(35);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x2);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x1);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(35);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x1);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(35);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x2);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x3 + (256L*x2) + (8960L*x1) + (313600L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_15 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-64L) + x1 + (64L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-64L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-64L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-64L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-64L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(224);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-128L) + x1 + (96L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-128L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-128L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-128L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-128L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(288);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-224L) + x1 + (64L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-224L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-224L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-224L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-224L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (288L*x0))] = tmp90;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (48L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(48L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(24L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)), static_cast<long>(48L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(24L); x2<static_cast<long>(25L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (25L*x1) + (25L*x1_inner) + (1200L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (48L*x2) + (1200L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_19 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(35L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(35L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(288L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(35);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-10368L) + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-10080L) + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x2);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-9792L) + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-288L) + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(288L + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x1);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(9792L + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(10080L + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(10368L + x3 + (288L*x2) + (10080L*x1) + (352800L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(36);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x1);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(35);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x2);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x1);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(35);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x1);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(35);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x2);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(35);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x2);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(35);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(35);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x2);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x1);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(35);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x2);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x1);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(35);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x1);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(35);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x2);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x3 + (288L*x2) + (10080L*x1) + (352800L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_convolution_max_pool2d_with_indices_20 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       const float* in_ptr20,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(128);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-64L) + x1 + (64L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-64L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-64L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-64L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-64L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(224);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-128L) + x1 + (96L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-128L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-128L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-128L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-128L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(288);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-224L) + x1 + (64L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-224L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-224L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-224L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-224L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (288L*x0))] = tmp90;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(16L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(288L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(576L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(10080L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp8, 8);
                            float tmp11[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(10368L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp11, 8);
                            float tmp14[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(10656L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp14, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(20160L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp17, 8);
                            float tmp20[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(20448L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp20, 8);
                            float tmp23[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(20736L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)), static_cast<long>(576L), tmp23, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(tmp11 + static_cast<long>(8L*x1_inner));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(tmp14 + static_cast<long>(8L*x1_inner));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x1_inner));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                                auto tmp24 = at::vec::Vectorized<float>::loadu(tmp23 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                auto tmp13 = at::vec::maximum(tmp12, tmp10);
                                auto tmp16 = at::vec::maximum(tmp15, tmp13);
                                auto tmp19 = at::vec::maximum(tmp18, tmp16);
                                auto tmp22 = at::vec::maximum(tmp21, tmp19);
                                auto tmp25 = at::vec::maximum(tmp24, tmp22);
                                tmp25.store(out_ptr1 + static_cast<long>(x3 + (17L*x2) + (289L*x1) + (289L*x1_inner) + (221952L*x0)));
                            }
                        }
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(16L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(288L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(576L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(10080L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(10368L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(10656L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(20160L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(20448L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(20736L + x1 + (576L*x3) + (20160L*x2) + (352800L*x0)));
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            auto tmp8 = at::vec::maximum(tmp7, tmp6);
                            auto tmp10 = at::vec::maximum(tmp9, tmp8);
                            auto tmp12 = at::vec::maximum(tmp11, tmp10);
                            auto tmp14 = at::vec::maximum(tmp13, tmp12);
                            auto tmp16 = at::vec::maximum(tmp15, tmp14);
                            { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp16.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (289L*x1_inner) + (221952L*x0))] = tmpbuf[x1_inner]; }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(288L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr20 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2592L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (288L*x2) + (2592L*x0)), static_cast<long>(288L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr20[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2592L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (288L*x2) + (2592L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(9800L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (110976L*x0)), static_cast<long>(384L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(0.001);
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(288L); x2<static_cast<long>(289L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (384L*x2) + (110976L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (96L*x2) + (27744L*x0)), static_cast<long>(96L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(0.001);
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr1 + static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(288L); x2<static_cast<long>(289L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (96L*x2) + (27744L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                        auto tmp2 = tmp0 - tmp1;
                        auto tmp4 = static_cast<float>(0.001);
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
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp17.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0))] = tmpbuf[x1_inner]; }
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(288L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp2 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)), static_cast<long>(768L));
                        at::vec::transpose_mxn<float,8,8>(tmp2, 8, out_ptr3 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)), static_cast<long>(768L));
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr4 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)), static_cast<long>(768L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(288L); x2<static_cast<long>(289L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr10[static_cast<long>(x2 + (289L*x1) + (289L*x1_inner) + (221952L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)));
                        tmp0.store(out_ptr3 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)));
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (768L*x2) + (221952L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (896L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (896L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(17L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + x2);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(17);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + x3);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-18L) + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr0[static_cast<long>((-17L) + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + x3);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr0[static_cast<long>((-16L) + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr0[static_cast<long>((-1L) + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr0[static_cast<long>(x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr0[static_cast<long>(1L + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + x2);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr0[static_cast<long>(16L + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr0[static_cast<long>(17L + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr0[static_cast<long>(18L + x3 + (17L*x2) + (289L*x1) + (221952L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(18);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + x2);
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(17);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + x3);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + x2);
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(17);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + x2);
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(17);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + x3);
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(17);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + x3);
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(17);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(17);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + x3);
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + x2);
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(17);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + x3);
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + x2);
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(17);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + x2);
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(17);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + x3);
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr0[static_cast<long>(x1 + (768L*x3) + (13056L*x2) + (221952L*x0))] = tmp251;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_31 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(384);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-192L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-192L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-192L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-192L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(576);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-384L) + x1 + (192L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-384L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-384L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-384L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-384L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(768);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-576L) + x1 + (192L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-576L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-576L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-576L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-576L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp90;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(17L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(17);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-13824L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-13056L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x2);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-12288L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-768L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(768L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x1);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(12288L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(13056L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(13824L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(18);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x1);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(17);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x2);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x1);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(17);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x1);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(17);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x2);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(17);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x2);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(17);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(17);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x2);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x1);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(17);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x2);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x1);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(17);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x1);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(17);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x2);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_39 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(384);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-192L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-192L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-192L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-192L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(576);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-384L) + x1 + (192L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-384L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-384L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-384L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-384L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(768);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-576L) + x1 + (192L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-576L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-576L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-576L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-576L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp90;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1120L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1120L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_46 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(17L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(17);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-13824L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-13056L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x2);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-12288L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-768L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(768L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x1);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(12288L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(13056L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(13824L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(18);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x1);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(17);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x2);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x1);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(17);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x1);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(17);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x2);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(17);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x2);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(17);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(17);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x2);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x1);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(17);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x2);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x1);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(17);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x1);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(17);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x2);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_47 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(384);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-192L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-192L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-192L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-192L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(576);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-384L) + x1 + (192L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-384L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-384L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-384L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-384L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(768);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-576L) + x1 + (192L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-576L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-576L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-576L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-576L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp90;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_54 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(17L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(768L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(17);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + x2);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-13824L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(in_ptr0 + static_cast<long>((-13056L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + x2);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(in_ptr0 + static_cast<long>((-12288L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(in_ptr0 + static_cast<long>((-768L) + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(in_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(in_ptr0 + static_cast<long>(768L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + x1);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(in_ptr0 + static_cast<long>(12288L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(in_ptr0 + static_cast<long>(13056L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(in_ptr0 + static_cast<long>(13824L + x3 + (768L*x2) + (13056L*x1) + (221952L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(18);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + x1);
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(17);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + x2);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp92 = tmp90 & tmp78;
                                auto tmp91 = [&]
                                {
                                    auto tmp93 = static_cast<float>(1.0);
                                    return tmp93;
                                }
                                ;
                                auto tmp94 = tmp90 ? tmp91() : static_cast<float>(1.0);
                                return tmp94;
                            }
                            ;
                            auto tmp95 = tmp78 ? tmp79() : static_cast<float>(0.0);
                            auto tmp96 = tmp14 >= tmp70;
                            auto tmp97 = tmp14 < tmp72;
                            auto tmp98 = tmp96 & tmp97;
                            auto tmp99 = tmp74 & tmp98;
                            auto tmp100 = [&]
                            {
                                auto tmp101 = c10::convert<int>((-1L) + x1);
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(17);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(x2);
                                auto tmp108 = tmp107 >= tmp102;
                                auto tmp109 = tmp107 < tmp104;
                                auto tmp110 = tmp108 & tmp109;
                                auto tmp111 = tmp106 & tmp110;
                                auto tmp113 = tmp111 & tmp99;
                                auto tmp112 = [&]
                                {
                                    auto tmp114 = static_cast<float>(1.0);
                                    return tmp114;
                                }
                                ;
                                auto tmp115 = tmp111 ? tmp112() : static_cast<float>(1.0);
                                return tmp115;
                            }
                            ;
                            auto tmp116 = tmp99 ? tmp100() : static_cast<float>(0.0);
                            auto tmp117 = decltype(tmp116)(tmp116 + tmp95);
                            auto tmp118 = tmp23 >= tmp70;
                            auto tmp119 = tmp23 < tmp72;
                            auto tmp120 = tmp118 & tmp119;
                            auto tmp121 = tmp74 & tmp120;
                            auto tmp122 = [&]
                            {
                                auto tmp123 = c10::convert<int>((-1L) + x1);
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(17);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + x2);
                                auto tmp130 = tmp129 >= tmp124;
                                auto tmp131 = tmp129 < tmp126;
                                auto tmp132 = tmp130 & tmp131;
                                auto tmp133 = tmp128 & tmp132;
                                auto tmp135 = tmp133 & tmp121;
                                auto tmp134 = [&]
                                {
                                    auto tmp136 = static_cast<float>(1.0);
                                    return tmp136;
                                }
                                ;
                                auto tmp137 = tmp133 ? tmp134() : static_cast<float>(1.0);
                                return tmp137;
                            }
                            ;
                            auto tmp138 = tmp121 ? tmp122() : static_cast<float>(0.0);
                            auto tmp139 = decltype(tmp138)(tmp138 + tmp117);
                            auto tmp140 = tmp32 >= tmp70;
                            auto tmp141 = tmp32 < tmp72;
                            auto tmp142 = tmp140 & tmp141;
                            auto tmp143 = tmp142 & tmp77;
                            auto tmp144 = [&]
                            {
                                auto tmp145 = c10::convert<int>(x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(17);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + x2);
                                auto tmp152 = tmp151 >= tmp146;
                                auto tmp153 = tmp151 < tmp148;
                                auto tmp154 = tmp152 & tmp153;
                                auto tmp155 = tmp150 & tmp154;
                                auto tmp157 = tmp155 & tmp143;
                                auto tmp156 = [&]
                                {
                                    auto tmp158 = static_cast<float>(1.0);
                                    return tmp158;
                                }
                                ;
                                auto tmp159 = tmp155 ? tmp156() : static_cast<float>(1.0);
                                return tmp159;
                            }
                            ;
                            auto tmp160 = tmp143 ? tmp144() : static_cast<float>(0.0);
                            auto tmp161 = decltype(tmp160)(tmp160 + tmp139);
                            auto tmp162 = tmp142 & tmp98;
                            auto tmp163 = [&]
                            {
                                auto tmp164 = c10::convert<int>(x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(17);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(x2);
                                auto tmp171 = tmp170 >= tmp165;
                                auto tmp172 = tmp170 < tmp167;
                                auto tmp173 = tmp171 & tmp172;
                                auto tmp174 = tmp169 & tmp173;
                                auto tmp176 = tmp174 & tmp162;
                                auto tmp175 = [&]
                                {
                                    auto tmp177 = static_cast<float>(1.0);
                                    return tmp177;
                                }
                                ;
                                auto tmp178 = tmp174 ? tmp175() : static_cast<float>(1.0);
                                return tmp178;
                            }
                            ;
                            auto tmp179 = tmp162 ? tmp163() : static_cast<float>(0.0);
                            auto tmp180 = decltype(tmp179)(tmp179 + tmp161);
                            auto tmp181 = tmp142 & tmp120;
                            auto tmp182 = [&]
                            {
                                auto tmp183 = c10::convert<int>(x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(17);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + x2);
                                auto tmp190 = tmp189 >= tmp184;
                                auto tmp191 = tmp189 < tmp186;
                                auto tmp192 = tmp190 & tmp191;
                                auto tmp193 = tmp188 & tmp192;
                                auto tmp195 = tmp193 & tmp181;
                                auto tmp194 = [&]
                                {
                                    auto tmp196 = static_cast<float>(1.0);
                                    return tmp196;
                                }
                                ;
                                auto tmp197 = tmp193 ? tmp194() : static_cast<float>(1.0);
                                return tmp197;
                            }
                            ;
                            auto tmp198 = tmp181 ? tmp182() : static_cast<float>(0.0);
                            auto tmp199 = decltype(tmp198)(tmp198 + tmp180);
                            auto tmp200 = tmp51 >= tmp70;
                            auto tmp201 = tmp51 < tmp72;
                            auto tmp202 = tmp200 & tmp201;
                            auto tmp203 = tmp202 & tmp77;
                            auto tmp204 = [&]
                            {
                                auto tmp205 = c10::convert<int>(1L + x1);
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(17);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + x2);
                                auto tmp212 = tmp211 >= tmp206;
                                auto tmp213 = tmp211 < tmp208;
                                auto tmp214 = tmp212 & tmp213;
                                auto tmp215 = tmp210 & tmp214;
                                auto tmp217 = tmp215 & tmp203;
                                auto tmp216 = [&]
                                {
                                    auto tmp218 = static_cast<float>(1.0);
                                    return tmp218;
                                }
                                ;
                                auto tmp219 = tmp215 ? tmp216() : static_cast<float>(1.0);
                                return tmp219;
                            }
                            ;
                            auto tmp220 = tmp203 ? tmp204() : static_cast<float>(0.0);
                            auto tmp221 = decltype(tmp220)(tmp220 + tmp199);
                            auto tmp222 = tmp202 & tmp98;
                            auto tmp223 = [&]
                            {
                                auto tmp224 = c10::convert<int>(1L + x1);
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(17);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(x2);
                                auto tmp231 = tmp230 >= tmp225;
                                auto tmp232 = tmp230 < tmp227;
                                auto tmp233 = tmp231 & tmp232;
                                auto tmp234 = tmp229 & tmp233;
                                auto tmp236 = tmp234 & tmp222;
                                auto tmp235 = [&]
                                {
                                    auto tmp237 = static_cast<float>(1.0);
                                    return tmp237;
                                }
                                ;
                                auto tmp238 = tmp234 ? tmp235() : static_cast<float>(1.0);
                                return tmp238;
                            }
                            ;
                            auto tmp239 = tmp222 ? tmp223() : static_cast<float>(0.0);
                            auto tmp240 = decltype(tmp239)(tmp239 + tmp221);
                            auto tmp241 = tmp202 & tmp120;
                            auto tmp242 = [&]
                            {
                                auto tmp243 = c10::convert<int>(1L + x1);
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(17);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + x2);
                                auto tmp250 = tmp249 >= tmp244;
                                auto tmp251 = tmp249 < tmp246;
                                auto tmp252 = tmp250 & tmp251;
                                auto tmp253 = tmp248 & tmp252;
                                auto tmp255 = tmp253 & tmp241;
                                auto tmp254 = [&]
                                {
                                    auto tmp256 = static_cast<float>(1.0);
                                    return tmp256;
                                }
                                ;
                                auto tmp257 = tmp253 ? tmp254() : static_cast<float>(1.0);
                                return tmp257;
                            }
                            ;
                            auto tmp258 = tmp241 ? tmp242() : static_cast<float>(0.0);
                            auto tmp259 = decltype(tmp258)(tmp258 + tmp240);
                            auto tmp260 = at::vec::Vectorized<float>(tmp259);
                            auto tmp261 = tmp69 / tmp260;
                            tmp261.store(out_ptr0 + static_cast<long>(x3 + (768L*x2) + (13056L*x1) + (221952L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_max_pool2d_with_indices_55 = async_compile.cpp('''
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
                       const float* in_ptr14,
                       const float* in_ptr15,
                       const float* in_ptr16,
                       const float* in_ptr17,
                       const float* in_ptr18,
                       const float* in_ptr19,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(192);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (192L*x0))];
                        auto tmp7 = in_ptr1[static_cast<long>(x1)];
                        auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                        auto tmp9 = in_ptr2[static_cast<long>(x1)];
                        auto tmp10 = static_cast<float>(0.001);
                        auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                        auto tmp12 = std::sqrt(tmp11);
                        auto tmp13 = 1 / tmp12;
                        auto tmp14 = static_cast<float>(1.0);
                        auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                        auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                        auto tmp17 = in_ptr3[static_cast<long>(x1)];
                        auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                        auto tmp19 = in_ptr4[static_cast<long>(x1)];
                        auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                        auto tmp21 = tmp20 * (tmp20>0);
                        return tmp21;
                    }
                    ;
                    auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp23 = tmp0 >= tmp3;
                    auto tmp24 = static_cast<long>(384);
                    auto tmp25 = tmp0 < tmp24;
                    auto tmp26 = tmp23 & tmp25;
                    auto tmp27 = [&]
                    {
                        auto tmp28 = in_ptr5[static_cast<long>((-192L) + x1 + (192L*x0))];
                        auto tmp29 = in_ptr6[static_cast<long>((-192L) + x1)];
                        auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                        auto tmp31 = in_ptr7[static_cast<long>((-192L) + x1)];
                        auto tmp32 = static_cast<float>(0.001);
                        auto tmp33 = decltype(tmp31)(tmp31 + tmp32);
                        auto tmp34 = std::sqrt(tmp33);
                        auto tmp35 = 1 / tmp34;
                        auto tmp36 = static_cast<float>(1.0);
                        auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                        auto tmp38 = decltype(tmp30)(tmp30 * tmp37);
                        auto tmp39 = in_ptr8[static_cast<long>((-192L) + x1)];
                        auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                        auto tmp41 = in_ptr9[static_cast<long>((-192L) + x1)];
                        auto tmp42 = decltype(tmp40)(tmp40 + tmp41);
                        auto tmp43 = tmp42 * (tmp42>0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp26 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                    auto tmp45 = tmp0 >= tmp24;
                    auto tmp46 = static_cast<long>(576);
                    auto tmp47 = tmp0 < tmp46;
                    auto tmp48 = tmp45 & tmp47;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = in_ptr10[static_cast<long>((-384L) + x1 + (192L*x0))];
                        auto tmp51 = in_ptr11[static_cast<long>((-384L) + x1)];
                        auto tmp52 = decltype(tmp50)(tmp50 - tmp51);
                        auto tmp53 = in_ptr12[static_cast<long>((-384L) + x1)];
                        auto tmp54 = static_cast<float>(0.001);
                        auto tmp55 = decltype(tmp53)(tmp53 + tmp54);
                        auto tmp56 = std::sqrt(tmp55);
                        auto tmp57 = 1 / tmp56;
                        auto tmp58 = static_cast<float>(1.0);
                        auto tmp59 = decltype(tmp57)(tmp57 * tmp58);
                        auto tmp60 = decltype(tmp52)(tmp52 * tmp59);
                        auto tmp61 = in_ptr13[static_cast<long>((-384L) + x1)];
                        auto tmp62 = decltype(tmp60)(tmp60 * tmp61);
                        auto tmp63 = in_ptr14[static_cast<long>((-384L) + x1)];
                        auto tmp64 = decltype(tmp62)(tmp62 + tmp63);
                        auto tmp65 = tmp64 * (tmp64>0);
                        return tmp65;
                    }
                    ;
                    auto tmp66 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp67 = tmp0 >= tmp46;
                    auto tmp68 = static_cast<long>(768);
                    auto tmp69 = tmp0 < tmp68;
                    auto tmp70 = [&]
                    {
                        auto tmp71 = in_ptr15[static_cast<long>((-576L) + x1 + (192L*x0))];
                        auto tmp72 = in_ptr16[static_cast<long>((-576L) + x1)];
                        auto tmp73 = decltype(tmp71)(tmp71 - tmp72);
                        auto tmp74 = in_ptr17[static_cast<long>((-576L) + x1)];
                        auto tmp75 = static_cast<float>(0.001);
                        auto tmp76 = decltype(tmp74)(tmp74 + tmp75);
                        auto tmp77 = std::sqrt(tmp76);
                        auto tmp78 = 1 / tmp77;
                        auto tmp79 = static_cast<float>(1.0);
                        auto tmp80 = decltype(tmp78)(tmp78 * tmp79);
                        auto tmp81 = decltype(tmp73)(tmp73 * tmp80);
                        auto tmp82 = in_ptr18[static_cast<long>((-576L) + x1)];
                        auto tmp83 = decltype(tmp81)(tmp81 * tmp82);
                        auto tmp84 = in_ptr19[static_cast<long>((-576L) + x1)];
                        auto tmp85 = decltype(tmp83)(tmp83 + tmp84);
                        auto tmp86 = tmp85 * (tmp85>0);
                        return tmp86;
                    }
                    ;
                    auto tmp87 = tmp67 ? tmp70() : static_cast<decltype(tmp70())>(0.0);
                    auto tmp88 = tmp48 ? tmp66 : tmp87;
                    auto tmp89 = tmp26 ? tmp44 : tmp88;
                    auto tmp90 = tmp4 ? tmp22 : tmp89;
                    out_ptr0[static_cast<long>(x1 + (768L*x0))] = tmp90;
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp0, 8);
                            float tmp2[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(768L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp2, 8);
                            float tmp5[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(1536L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp5, 8);
                            float tmp8[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(13056L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp8, 8);
                            float tmp11[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(13824L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp11, 8);
                            float tmp14[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(14592L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp14, 8);
                            float tmp17[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(26112L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp17, 8);
                            float tmp20[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(26880L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp20, 8);
                            float tmp23[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(27648L + x1 + (1536L*x3) + (26112L*x2) + (221952L*x0)), static_cast<long>(1536L), tmp23, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp3 = at::vec::Vectorized<float>::loadu(tmp2 + static_cast<long>(8L*x1_inner));
                                auto tmp6 = at::vec::Vectorized<float>::loadu(tmp5 + static_cast<long>(8L*x1_inner));
                                auto tmp9 = at::vec::Vectorized<float>::loadu(tmp8 + static_cast<long>(8L*x1_inner));
                                auto tmp12 = at::vec::Vectorized<float>::loadu(tmp11 + static_cast<long>(8L*x1_inner));
                                auto tmp15 = at::vec::Vectorized<float>::loadu(tmp14 + static_cast<long>(8L*x1_inner));
                                auto tmp18 = at::vec::Vectorized<float>::loadu(tmp17 + static_cast<long>(8L*x1_inner));
                                auto tmp21 = at::vec::Vectorized<float>::loadu(tmp20 + static_cast<long>(8L*x1_inner));
                                auto tmp24 = at::vec::Vectorized<float>::loadu(tmp23 + static_cast<long>(8L*x1_inner));
                                auto tmp4 = at::vec::maximum(tmp3, tmp1);
                                auto tmp7 = at::vec::maximum(tmp6, tmp4);
                                auto tmp10 = at::vec::maximum(tmp9, tmp7);
                                auto tmp13 = at::vec::maximum(tmp12, tmp10);
                                auto tmp16 = at::vec::maximum(tmp15, tmp13);
                                auto tmp19 = at::vec::maximum(tmp18, tmp16);
                                auto tmp22 = at::vec::maximum(tmp21, tmp19);
                                auto tmp25 = at::vec::maximum(tmp24, tmp22);
                                tmp25.store(out_ptr1 + static_cast<long>(x3 + (8L*x2) + (64L*x1) + (64L*x1_inner) + (81920L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(320L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (7L*x1) + (7L*x1_inner) + (1344L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1344L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2312L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_60 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (20480L*x0)), static_cast<long>(320L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(0.001);
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (81920L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (192L*x2) + (12288L*x0)), static_cast<long>(192L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                                auto tmp14 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                                auto tmp17 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = static_cast<float>(0.001);
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
                                auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                                tmp20.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (81920L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (81920L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp2 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (1280L*x2) + (81920L*x0)), static_cast<long>(1280L));
                        at::vec::transpose_mxn<float,8,8>(tmp2, 8, out_ptr3 + static_cast<long>(x1 + (1280L*x2) + (81920L*x0)), static_cast<long>(1280L));
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr4 + static_cast<long>(x1 + (1280L*x2) + (81920L*x0)), static_cast<long>(1280L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_61 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_63 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(384);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (24576L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(0.001);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(768);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-384L) + x2 + (384L*x1) + (24576L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-384L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-384L) + x2)];
                            auto tmp31 = static_cast<float>(0.001);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-384L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-384L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (64L*x2) + (131072L*x0))] = tmp44;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4032L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (448L*x2) + (4032L*x0)), static_cast<long>(448L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4032L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x2) + (4032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_66 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_cat_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(384);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (24576L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(0.001);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(768);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-384L) + x2 + (384L*x1) + (24576L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-384L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-384L) + x2)];
                            auto tmp31 = static_cast<float>(0.001);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-384L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-384L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (64L*x2) + (131072L*x0))] = tmp44;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1280L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + x2);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + x3);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr10[static_cast<long>((-9L) + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr10[static_cast<long>((-8L) + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + x3);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr10[static_cast<long>((-7L) + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr10[static_cast<long>((-1L) + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr10[static_cast<long>(x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr10[static_cast<long>(1L + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + x2);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr10[static_cast<long>(7L + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr10[static_cast<long>(8L + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr10[static_cast<long>(9L + x3 + (8L*x2) + (64L*x1) + (81920L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(9);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + x2);
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(8);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + x3);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + x2);
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(8);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + x2);
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(8);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + x3);
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(8);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + x3);
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(8);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(8);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + x3);
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + x2);
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(8);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + x3);
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + x2);
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(8);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + x2);
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(8);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + x3);
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr1[static_cast<long>(x1 + (1280L*x3) + (10240L*x2) + (81920L*x0))] = tmp251;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (20480L*x0)), static_cast<long>(320L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(0.001);
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (192L*x2) + (12288L*x0)), static_cast<long>(192L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                                auto tmp14 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                                auto tmp17 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = static_cast<float>(0.001);
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
                                auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                                tmp20.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        float tmp2[8*8] __attribute__ ((aligned (8)));
                        float tmp3[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp2 + static_cast<long>(8L*x1_inner));
                            tmp0.store(tmp3 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)), static_cast<long>(2048L));
                        at::vec::transpose_mxn<float,8,8>(tmp2, 8, out_ptr3 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)), static_cast<long>(2048L));
                        at::vec::transpose_mxn<float,8,8>(tmp3, 8, out_ptr4 + static_cast<long>(x1 + (2048L*x2) + (131072L*x0)), static_cast<long>(2048L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_cat_71 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(384);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (24576L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(0.001);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(768);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-384L) + x2 + (384L*x1) + (24576L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-384L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-384L) + x2)];
                            auto tmp31 = static_cast<float>(0.001);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-384L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-384L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (64L*x2) + (131072L*x0))] = tmp44;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4032L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (448L*x2) + (4032L*x0)), static_cast<long>(448L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4032L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (448L*x2) + (4032L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = static_cast<float>(0.001);
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (3L*x1) + (3L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_cat_75 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(0);
                        auto tmp2 = tmp0 >= tmp1;
                        auto tmp3 = static_cast<long>(384);
                        auto tmp4 = tmp0 < tmp3;
                        auto tmp5 = [&]
                        {
                            auto tmp6 = in_ptr0[static_cast<long>(x2 + (384L*x1) + (24576L*x0))];
                            auto tmp7 = in_ptr1[static_cast<long>(x2)];
                            auto tmp8 = decltype(tmp6)(tmp6 - tmp7);
                            auto tmp9 = in_ptr2[static_cast<long>(x2)];
                            auto tmp10 = static_cast<float>(0.001);
                            auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                            auto tmp12 = std::sqrt(tmp11);
                            auto tmp13 = 1 / tmp12;
                            auto tmp14 = static_cast<float>(1.0);
                            auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                            auto tmp16 = decltype(tmp8)(tmp8 * tmp15);
                            auto tmp17 = in_ptr3[static_cast<long>(x2)];
                            auto tmp18 = decltype(tmp16)(tmp16 * tmp17);
                            auto tmp19 = in_ptr4[static_cast<long>(x2)];
                            auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                            auto tmp21 = tmp20 * (tmp20>0);
                            return tmp21;
                        }
                        ;
                        auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                        auto tmp23 = tmp0 >= tmp3;
                        auto tmp24 = static_cast<long>(768);
                        auto tmp25 = tmp0 < tmp24;
                        auto tmp26 = [&]
                        {
                            auto tmp27 = in_ptr5[static_cast<long>((-384L) + x2 + (384L*x1) + (24576L*x0))];
                            auto tmp28 = in_ptr6[static_cast<long>((-384L) + x2)];
                            auto tmp29 = decltype(tmp27)(tmp27 - tmp28);
                            auto tmp30 = in_ptr7[static_cast<long>((-384L) + x2)];
                            auto tmp31 = static_cast<float>(0.001);
                            auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                            auto tmp33 = std::sqrt(tmp32);
                            auto tmp34 = 1 / tmp33;
                            auto tmp35 = static_cast<float>(1.0);
                            auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                            auto tmp37 = decltype(tmp29)(tmp29 * tmp36);
                            auto tmp38 = in_ptr8[static_cast<long>((-384L) + x2)];
                            auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                            auto tmp40 = in_ptr9[static_cast<long>((-384L) + x2)];
                            auto tmp41 = decltype(tmp39)(tmp39 + tmp40);
                            auto tmp42 = tmp41 * (tmp41>0);
                            return tmp42;
                        }
                        ;
                        auto tmp43 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                        auto tmp44 = tmp4 ? tmp22 : tmp43;
                        out_ptr0[static_cast<long>(x1 + (64L*x2) + (131072L*x0))] = tmp44;
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(8L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + x2);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(8);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + x3);
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr10[static_cast<long>((-9L) + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            auto tmp14 = c10::convert<long>(x3);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = in_ptr10[static_cast<long>((-8L) + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                            auto tmp22 = decltype(tmp21)(tmp21 + tmp13);
                            auto tmp23 = c10::convert<long>(1L + x3);
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = in_ptr10[static_cast<long>((-7L) + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : static_cast<decltype(tmp28())>(0.0);
                            auto tmp31 = decltype(tmp30)(tmp30 + tmp22);
                            auto tmp32 = c10::convert<long>(x2);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = in_ptr10[static_cast<long>((-1L) + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : static_cast<decltype(tmp37())>(0.0);
                            auto tmp40 = decltype(tmp39)(tmp39 + tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = in_ptr10[static_cast<long>(x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                            auto tmp45 = decltype(tmp44)(tmp44 + tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = in_ptr10[static_cast<long>(1L + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : static_cast<decltype(tmp47())>(0.0);
                            auto tmp50 = decltype(tmp49)(tmp49 + tmp45);
                            auto tmp51 = c10::convert<long>(1L + x2);
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = in_ptr10[static_cast<long>(7L + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : static_cast<decltype(tmp56())>(0.0);
                            auto tmp59 = decltype(tmp58)(tmp58 + tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = in_ptr10[static_cast<long>(8L + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : static_cast<decltype(tmp61())>(0.0);
                            auto tmp64 = decltype(tmp63)(tmp63 + tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = in_ptr10[static_cast<long>(9L + x3 + (8L*x2) + (64L*x1) + (131072L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : static_cast<decltype(tmp66())>(0.0);
                            auto tmp69 = decltype(tmp68)(tmp68 + tmp64);
                            auto tmp70 = static_cast<long>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<long>(9);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<long>((-1L) + x2);
                                auto tmp81 = static_cast<long>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<long>(8);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<long>((-1L) + x3);
                                auto tmp87 = tmp86 >= tmp81;
                                auto tmp88 = tmp86 < tmp83;
                                auto tmp89 = tmp87 & tmp88;
                                auto tmp90 = tmp85 & tmp89;
                                auto tmp91 = [&]
                                {
                                    auto tmp92 = static_cast<float>(1.0);
                                    return tmp92;
                                }
                                ;
                                auto tmp93 = tmp90 ? tmp91() : static_cast<decltype(tmp91())>(1.0);
                                return tmp93;
                            }
                            ;
                            auto tmp94 = tmp78 ? tmp79() : static_cast<decltype(tmp79())>(0.0);
                            auto tmp95 = tmp14 >= tmp70;
                            auto tmp96 = tmp14 < tmp72;
                            auto tmp97 = tmp95 & tmp96;
                            auto tmp98 = tmp74 & tmp97;
                            auto tmp99 = [&]
                            {
                                auto tmp100 = c10::convert<long>((-1L) + x2);
                                auto tmp101 = static_cast<long>(0);
                                auto tmp102 = tmp100 >= tmp101;
                                auto tmp103 = static_cast<long>(8);
                                auto tmp104 = tmp100 < tmp103;
                                auto tmp105 = tmp102 & tmp104;
                                auto tmp106 = c10::convert<long>(x3);
                                auto tmp107 = tmp106 >= tmp101;
                                auto tmp108 = tmp106 < tmp103;
                                auto tmp109 = tmp107 & tmp108;
                                auto tmp110 = tmp105 & tmp109;
                                auto tmp111 = [&]
                                {
                                    auto tmp112 = static_cast<float>(1.0);
                                    return tmp112;
                                }
                                ;
                                auto tmp113 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(1.0);
                                return tmp113;
                            }
                            ;
                            auto tmp114 = tmp98 ? tmp99() : static_cast<decltype(tmp99())>(0.0);
                            auto tmp115 = decltype(tmp114)(tmp114 + tmp94);
                            auto tmp116 = tmp23 >= tmp70;
                            auto tmp117 = tmp23 < tmp72;
                            auto tmp118 = tmp116 & tmp117;
                            auto tmp119 = tmp74 & tmp118;
                            auto tmp120 = [&]
                            {
                                auto tmp121 = c10::convert<long>((-1L) + x2);
                                auto tmp122 = static_cast<long>(0);
                                auto tmp123 = tmp121 >= tmp122;
                                auto tmp124 = static_cast<long>(8);
                                auto tmp125 = tmp121 < tmp124;
                                auto tmp126 = tmp123 & tmp125;
                                auto tmp127 = c10::convert<long>(1L + x3);
                                auto tmp128 = tmp127 >= tmp122;
                                auto tmp129 = tmp127 < tmp124;
                                auto tmp130 = tmp128 & tmp129;
                                auto tmp131 = tmp126 & tmp130;
                                auto tmp132 = [&]
                                {
                                    auto tmp133 = static_cast<float>(1.0);
                                    return tmp133;
                                }
                                ;
                                auto tmp134 = tmp131 ? tmp132() : static_cast<decltype(tmp132())>(1.0);
                                return tmp134;
                            }
                            ;
                            auto tmp135 = tmp119 ? tmp120() : static_cast<decltype(tmp120())>(0.0);
                            auto tmp136 = decltype(tmp135)(tmp135 + tmp115);
                            auto tmp137 = tmp32 >= tmp70;
                            auto tmp138 = tmp32 < tmp72;
                            auto tmp139 = tmp137 & tmp138;
                            auto tmp140 = tmp139 & tmp77;
                            auto tmp141 = [&]
                            {
                                auto tmp142 = c10::convert<long>(x2);
                                auto tmp143 = static_cast<long>(0);
                                auto tmp144 = tmp142 >= tmp143;
                                auto tmp145 = static_cast<long>(8);
                                auto tmp146 = tmp142 < tmp145;
                                auto tmp147 = tmp144 & tmp146;
                                auto tmp148 = c10::convert<long>((-1L) + x3);
                                auto tmp149 = tmp148 >= tmp143;
                                auto tmp150 = tmp148 < tmp145;
                                auto tmp151 = tmp149 & tmp150;
                                auto tmp152 = tmp147 & tmp151;
                                auto tmp153 = [&]
                                {
                                    auto tmp154 = static_cast<float>(1.0);
                                    return tmp154;
                                }
                                ;
                                auto tmp155 = tmp152 ? tmp153() : static_cast<decltype(tmp153())>(1.0);
                                return tmp155;
                            }
                            ;
                            auto tmp156 = tmp140 ? tmp141() : static_cast<decltype(tmp141())>(0.0);
                            auto tmp157 = decltype(tmp156)(tmp156 + tmp136);
                            auto tmp158 = tmp139 & tmp97;
                            auto tmp159 = [&]
                            {
                                auto tmp160 = c10::convert<long>(x2);
                                auto tmp161 = static_cast<long>(0);
                                auto tmp162 = tmp160 >= tmp161;
                                auto tmp163 = static_cast<long>(8);
                                auto tmp164 = tmp160 < tmp163;
                                auto tmp165 = tmp162 & tmp164;
                                auto tmp166 = c10::convert<long>(x3);
                                auto tmp167 = tmp166 >= tmp161;
                                auto tmp168 = tmp166 < tmp163;
                                auto tmp169 = tmp167 & tmp168;
                                auto tmp170 = tmp165 & tmp169;
                                auto tmp171 = [&]
                                {
                                    auto tmp172 = static_cast<float>(1.0);
                                    return tmp172;
                                }
                                ;
                                auto tmp173 = tmp170 ? tmp171() : static_cast<decltype(tmp171())>(1.0);
                                return tmp173;
                            }
                            ;
                            auto tmp174 = tmp158 ? tmp159() : static_cast<decltype(tmp159())>(0.0);
                            auto tmp175 = decltype(tmp174)(tmp174 + tmp157);
                            auto tmp176 = tmp139 & tmp118;
                            auto tmp177 = [&]
                            {
                                auto tmp178 = c10::convert<long>(x2);
                                auto tmp179 = static_cast<long>(0);
                                auto tmp180 = tmp178 >= tmp179;
                                auto tmp181 = static_cast<long>(8);
                                auto tmp182 = tmp178 < tmp181;
                                auto tmp183 = tmp180 & tmp182;
                                auto tmp184 = c10::convert<long>(1L + x3);
                                auto tmp185 = tmp184 >= tmp179;
                                auto tmp186 = tmp184 < tmp181;
                                auto tmp187 = tmp185 & tmp186;
                                auto tmp188 = tmp183 & tmp187;
                                auto tmp189 = [&]
                                {
                                    auto tmp190 = static_cast<float>(1.0);
                                    return tmp190;
                                }
                                ;
                                auto tmp191 = tmp188 ? tmp189() : static_cast<decltype(tmp189())>(1.0);
                                return tmp191;
                            }
                            ;
                            auto tmp192 = tmp176 ? tmp177() : static_cast<decltype(tmp177())>(0.0);
                            auto tmp193 = decltype(tmp192)(tmp192 + tmp175);
                            auto tmp194 = tmp51 >= tmp70;
                            auto tmp195 = tmp51 < tmp72;
                            auto tmp196 = tmp194 & tmp195;
                            auto tmp197 = tmp196 & tmp77;
                            auto tmp198 = [&]
                            {
                                auto tmp199 = c10::convert<long>(1L + x2);
                                auto tmp200 = static_cast<long>(0);
                                auto tmp201 = tmp199 >= tmp200;
                                auto tmp202 = static_cast<long>(8);
                                auto tmp203 = tmp199 < tmp202;
                                auto tmp204 = tmp201 & tmp203;
                                auto tmp205 = c10::convert<long>((-1L) + x3);
                                auto tmp206 = tmp205 >= tmp200;
                                auto tmp207 = tmp205 < tmp202;
                                auto tmp208 = tmp206 & tmp207;
                                auto tmp209 = tmp204 & tmp208;
                                auto tmp210 = [&]
                                {
                                    auto tmp211 = static_cast<float>(1.0);
                                    return tmp211;
                                }
                                ;
                                auto tmp212 = tmp209 ? tmp210() : static_cast<decltype(tmp210())>(1.0);
                                return tmp212;
                            }
                            ;
                            auto tmp213 = tmp197 ? tmp198() : static_cast<decltype(tmp198())>(0.0);
                            auto tmp214 = decltype(tmp213)(tmp213 + tmp193);
                            auto tmp215 = tmp196 & tmp97;
                            auto tmp216 = [&]
                            {
                                auto tmp217 = c10::convert<long>(1L + x2);
                                auto tmp218 = static_cast<long>(0);
                                auto tmp219 = tmp217 >= tmp218;
                                auto tmp220 = static_cast<long>(8);
                                auto tmp221 = tmp217 < tmp220;
                                auto tmp222 = tmp219 & tmp221;
                                auto tmp223 = c10::convert<long>(x3);
                                auto tmp224 = tmp223 >= tmp218;
                                auto tmp225 = tmp223 < tmp220;
                                auto tmp226 = tmp224 & tmp225;
                                auto tmp227 = tmp222 & tmp226;
                                auto tmp228 = [&]
                                {
                                    auto tmp229 = static_cast<float>(1.0);
                                    return tmp229;
                                }
                                ;
                                auto tmp230 = tmp227 ? tmp228() : static_cast<decltype(tmp228())>(1.0);
                                return tmp230;
                            }
                            ;
                            auto tmp231 = tmp215 ? tmp216() : static_cast<decltype(tmp216())>(0.0);
                            auto tmp232 = decltype(tmp231)(tmp231 + tmp214);
                            auto tmp233 = tmp196 & tmp118;
                            auto tmp234 = [&]
                            {
                                auto tmp235 = c10::convert<long>(1L + x2);
                                auto tmp236 = static_cast<long>(0);
                                auto tmp237 = tmp235 >= tmp236;
                                auto tmp238 = static_cast<long>(8);
                                auto tmp239 = tmp235 < tmp238;
                                auto tmp240 = tmp237 & tmp239;
                                auto tmp241 = c10::convert<long>(1L + x3);
                                auto tmp242 = tmp241 >= tmp236;
                                auto tmp243 = tmp241 < tmp238;
                                auto tmp244 = tmp242 & tmp243;
                                auto tmp245 = tmp240 & tmp244;
                                auto tmp246 = [&]
                                {
                                    auto tmp247 = static_cast<float>(1.0);
                                    return tmp247;
                                }
                                ;
                                auto tmp248 = tmp245 ? tmp246() : static_cast<decltype(tmp246())>(1.0);
                                return tmp248;
                            }
                            ;
                            auto tmp249 = tmp233 ? tmp234() : static_cast<decltype(tmp234())>(0.0);
                            auto tmp250 = decltype(tmp249)(tmp249 + tmp232);
                            auto tmp251 = tmp69 / tmp250;
                            out_ptr1[static_cast<long>(x1 + (2048L*x3) + (16384L*x2) + (131072L*x0))] = tmp251;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_76 = async_compile.cpp('''
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
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       float* out_ptr0,
                       float* out_ptr1)
{
    auto out_ptr2 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(320L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (320L*x2) + (20480L*x0)), static_cast<long>(320L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 - tmp3;
                            auto tmp6 = static_cast<float>(0.001);
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                        {
                            float tmp0[8*8] __attribute__ ((aligned (8)));
                            at::vec::transpose_mxn<float,8,8>(in_ptr5 + static_cast<long>(x1 + (192L*x2) + (12288L*x0)), static_cast<long>(192L), tmp0, 8);
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                                auto tmp2 = in_ptr6[static_cast<long>(x1 + x1_inner)];
                                auto tmp5 = in_ptr7[static_cast<long>(x1 + x1_inner)];
                                auto tmp14 = in_ptr8[static_cast<long>(x1 + x1_inner)];
                                auto tmp17 = in_ptr9[static_cast<long>(x1 + x1_inner)];
                                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                                auto tmp4 = tmp1 - tmp3;
                                auto tmp6 = static_cast<float>(0.001);
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
                                auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                                tmp20.store(out_ptr1 + static_cast<long>(x2 + (64L*x1) + (64L*x1_inner) + (131072L*x0)));
                            }
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1 + (64L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr2[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16384L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(64.0);
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (80, ), (1, ))
    assert_size_stride(arg7_1, (80, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (48, ), (1, ))
    assert_size_stride(arg13_1, (48, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (96, ), (1, ))
    assert_size_stride(arg19_1, (96, ), (1, ))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (32, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (48, ), (1, ))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (48, ), (1, ))
    assert_size_stride(arg41_1, (48, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (96, ), (1, ))
    assert_size_stride(arg47_1, (96, ), (1, ))
    assert_size_stride(arg48_1, (96, ), (1, ))
    assert_size_stride(arg49_1, (96, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, ), (1, ))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (96, ), (1, ))
    assert_size_stride(arg57_1, (96, ), (1, ))
    assert_size_stride(arg58_1, (96, ), (1, ))
    assert_size_stride(arg59_1, (96, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (192, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (192, ), (1, ))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (192, ), (1, ))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (192, ), (1, ))
    assert_size_stride(arg81_1, (192, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (160, ), (1, ))
    assert_size_stride(arg85_1, (160, ), (1, ))
    assert_size_stride(arg86_1, (192, ), (1, ))
    assert_size_stride(arg87_1, (192, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (160, ), (1, ))
    assert_size_stride(arg91_1, (160, ), (1, ))
    assert_size_stride(arg92_1, (160, ), (1, ))
    assert_size_stride(arg93_1, (160, ), (1, ))
    assert_size_stride(arg94_1, (160, ), (1, ))
    assert_size_stride(arg95_1, (160, ), (1, ))
    assert_size_stride(arg96_1, (192, ), (1, ))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, ), (1, ))
    assert_size_stride(arg102_1, (160, ), (1, ))
    assert_size_stride(arg103_1, (160, ), (1, ))
    assert_size_stride(arg104_1, (160, ), (1, ))
    assert_size_stride(arg105_1, (160, ), (1, ))
    assert_size_stride(arg106_1, (192, ), (1, ))
    assert_size_stride(arg107_1, (192, ), (1, ))
    assert_size_stride(arg108_1, (160, ), (1, ))
    assert_size_stride(arg109_1, (160, ), (1, ))
    assert_size_stride(arg110_1, (160, ), (1, ))
    assert_size_stride(arg111_1, (160, ), (1, ))
    assert_size_stride(arg112_1, (160, ), (1, ))
    assert_size_stride(arg113_1, (160, ), (1, ))
    assert_size_stride(arg114_1, (160, ), (1, ))
    assert_size_stride(arg115_1, (160, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (192, ), (1, ))
    assert_size_stride(arg118_1, (192, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (192, ), (1, ))
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (192, ), (1, ))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (192, ), (1, ))
    assert_size_stride(arg130_1, (192, ), (1, ))
    assert_size_stride(arg131_1, (192, ), (1, ))
    assert_size_stride(arg132_1, (192, ), (1, ))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (192, ), (1, ))
    assert_size_stride(arg136_1, (192, ), (1, ))
    assert_size_stride(arg137_1, (192, ), (1, ))
    assert_size_stride(arg138_1, (192, ), (1, ))
    assert_size_stride(arg139_1, (192, ), (1, ))
    assert_size_stride(arg140_1, (192, ), (1, ))
    assert_size_stride(arg141_1, (192, ), (1, ))
    assert_size_stride(arg142_1, (320, ), (1, ))
    assert_size_stride(arg143_1, (320, ), (1, ))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (192, ), (1, ))
    assert_size_stride(arg148_1, (192, ), (1, ))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (320, ), (1, ))
    assert_size_stride(arg153_1, (320, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (448, ), (1, ))
    assert_size_stride(arg161_1, (448, ), (1, ))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (192, ), (1, ))
    assert_size_stride(arg169_1, (192, ), (1, ))
    assert_size_stride(arg170_1, (320, ), (1, ))
    assert_size_stride(arg171_1, (320, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (448, ), (1, ))
    assert_size_stride(arg179_1, (448, ), (1, ))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg189_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg190_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg191_1, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg192_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg193_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg194_1, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg195_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg196_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg197_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg198_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg199_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg200_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg201_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg202_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg203_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg204_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg205_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg206_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg207_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg208_1, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg209_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg210_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg211_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg212_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg213_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg214_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg215_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg216_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg217_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg218_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg219_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg220_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg221_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg222_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg223_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg224_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg225_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg226_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg227_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg228_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg229_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg230_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg231_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg232_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg233_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg234_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg235_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg236_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg237_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg238_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg239_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg240_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg241_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg242_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg243_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg244_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg245_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg246_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg247_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg248_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg249_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg250_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg251_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg252_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg253_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg254_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg255_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg256_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg257_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg258_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg259_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg260_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg261_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg262_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg263_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg264_1, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg265_1, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg266_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg267_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg268_1, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg269_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg270_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg271_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg272_1, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg273_1, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg274_1, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg275_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg276_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg277_1, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg278_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg279_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg280_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg281_1, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg282_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg283_1, (1000, ), (1, ))
    assert_size_stride(arg284_1, (32, ), (1, ))
    assert_size_stride(arg285_1, (32, ), (1, ))
    assert_size_stride(arg286_1, (32, ), (1, ))
    assert_size_stride(arg287_1, (32, ), (1, ))
    assert_size_stride(arg288_1, (64, ), (1, ))
    assert_size_stride(arg289_1, (64, ), (1, ))
    assert_size_stride(arg290_1, (80, ), (1, ))
    assert_size_stride(arg291_1, (80, ), (1, ))
    assert_size_stride(arg292_1, (192, ), (1, ))
    assert_size_stride(arg293_1, (192, ), (1, ))
    assert_size_stride(arg294_1, (64, ), (1, ))
    assert_size_stride(arg295_1, (64, ), (1, ))
    assert_size_stride(arg296_1, (48, ), (1, ))
    assert_size_stride(arg297_1, (48, ), (1, ))
    assert_size_stride(arg298_1, (64, ), (1, ))
    assert_size_stride(arg299_1, (64, ), (1, ))
    assert_size_stride(arg300_1, (64, ), (1, ))
    assert_size_stride(arg301_1, (64, ), (1, ))
    assert_size_stride(arg302_1, (96, ), (1, ))
    assert_size_stride(arg303_1, (96, ), (1, ))
    assert_size_stride(arg304_1, (96, ), (1, ))
    assert_size_stride(arg305_1, (96, ), (1, ))
    assert_size_stride(arg306_1, (32, ), (1, ))
    assert_size_stride(arg307_1, (32, ), (1, ))
    assert_size_stride(arg308_1, (64, ), (1, ))
    assert_size_stride(arg309_1, (64, ), (1, ))
    assert_size_stride(arg310_1, (48, ), (1, ))
    assert_size_stride(arg311_1, (48, ), (1, ))
    assert_size_stride(arg312_1, (64, ), (1, ))
    assert_size_stride(arg313_1, (64, ), (1, ))
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (96, ), (1, ))
    assert_size_stride(arg317_1, (96, ), (1, ))
    assert_size_stride(arg318_1, (96, ), (1, ))
    assert_size_stride(arg319_1, (96, ), (1, ))
    assert_size_stride(arg320_1, (64, ), (1, ))
    assert_size_stride(arg321_1, (64, ), (1, ))
    assert_size_stride(arg322_1, (64, ), (1, ))
    assert_size_stride(arg323_1, (64, ), (1, ))
    assert_size_stride(arg324_1, (48, ), (1, ))
    assert_size_stride(arg325_1, (48, ), (1, ))
    assert_size_stride(arg326_1, (64, ), (1, ))
    assert_size_stride(arg327_1, (64, ), (1, ))
    assert_size_stride(arg328_1, (64, ), (1, ))
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (96, ), (1, ))
    assert_size_stride(arg331_1, (96, ), (1, ))
    assert_size_stride(arg332_1, (96, ), (1, ))
    assert_size_stride(arg333_1, (96, ), (1, ))
    assert_size_stride(arg334_1, (64, ), (1, ))
    assert_size_stride(arg335_1, (64, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (384, ), (1, ))
    assert_size_stride(arg338_1, (64, ), (1, ))
    assert_size_stride(arg339_1, (64, ), (1, ))
    assert_size_stride(arg340_1, (96, ), (1, ))
    assert_size_stride(arg341_1, (96, ), (1, ))
    assert_size_stride(arg342_1, (96, ), (1, ))
    assert_size_stride(arg343_1, (96, ), (1, ))
    assert_size_stride(arg344_1, (192, ), (1, ))
    assert_size_stride(arg345_1, (192, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (128, ), (1, ))
    assert_size_stride(arg348_1, (128, ), (1, ))
    assert_size_stride(arg349_1, (128, ), (1, ))
    assert_size_stride(arg350_1, (192, ), (1, ))
    assert_size_stride(arg351_1, (192, ), (1, ))
    assert_size_stride(arg352_1, (128, ), (1, ))
    assert_size_stride(arg353_1, (128, ), (1, ))
    assert_size_stride(arg354_1, (128, ), (1, ))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (128, ), (1, ))
    assert_size_stride(arg360_1, (192, ), (1, ))
    assert_size_stride(arg361_1, (192, ), (1, ))
    assert_size_stride(arg362_1, (192, ), (1, ))
    assert_size_stride(arg363_1, (192, ), (1, ))
    assert_size_stride(arg364_1, (192, ), (1, ))
    assert_size_stride(arg365_1, (192, ), (1, ))
    assert_size_stride(arg366_1, (160, ), (1, ))
    assert_size_stride(arg367_1, (160, ), (1, ))
    assert_size_stride(arg368_1, (160, ), (1, ))
    assert_size_stride(arg369_1, (160, ), (1, ))
    assert_size_stride(arg370_1, (192, ), (1, ))
    assert_size_stride(arg371_1, (192, ), (1, ))
    assert_size_stride(arg372_1, (160, ), (1, ))
    assert_size_stride(arg373_1, (160, ), (1, ))
    assert_size_stride(arg374_1, (160, ), (1, ))
    assert_size_stride(arg375_1, (160, ), (1, ))
    assert_size_stride(arg376_1, (160, ), (1, ))
    assert_size_stride(arg377_1, (160, ), (1, ))
    assert_size_stride(arg378_1, (160, ), (1, ))
    assert_size_stride(arg379_1, (160, ), (1, ))
    assert_size_stride(arg380_1, (192, ), (1, ))
    assert_size_stride(arg381_1, (192, ), (1, ))
    assert_size_stride(arg382_1, (192, ), (1, ))
    assert_size_stride(arg383_1, (192, ), (1, ))
    assert_size_stride(arg384_1, (192, ), (1, ))
    assert_size_stride(arg385_1, (192, ), (1, ))
    assert_size_stride(arg386_1, (160, ), (1, ))
    assert_size_stride(arg387_1, (160, ), (1, ))
    assert_size_stride(arg388_1, (160, ), (1, ))
    assert_size_stride(arg389_1, (160, ), (1, ))
    assert_size_stride(arg390_1, (192, ), (1, ))
    assert_size_stride(arg391_1, (192, ), (1, ))
    assert_size_stride(arg392_1, (160, ), (1, ))
    assert_size_stride(arg393_1, (160, ), (1, ))
    assert_size_stride(arg394_1, (160, ), (1, ))
    assert_size_stride(arg395_1, (160, ), (1, ))
    assert_size_stride(arg396_1, (160, ), (1, ))
    assert_size_stride(arg397_1, (160, ), (1, ))
    assert_size_stride(arg398_1, (160, ), (1, ))
    assert_size_stride(arg399_1, (160, ), (1, ))
    assert_size_stride(arg400_1, (192, ), (1, ))
    assert_size_stride(arg401_1, (192, ), (1, ))
    assert_size_stride(arg402_1, (192, ), (1, ))
    assert_size_stride(arg403_1, (192, ), (1, ))
    assert_size_stride(arg404_1, (192, ), (1, ))
    assert_size_stride(arg405_1, (192, ), (1, ))
    assert_size_stride(arg406_1, (192, ), (1, ))
    assert_size_stride(arg407_1, (192, ), (1, ))
    assert_size_stride(arg408_1, (192, ), (1, ))
    assert_size_stride(arg409_1, (192, ), (1, ))
    assert_size_stride(arg410_1, (192, ), (1, ))
    assert_size_stride(arg411_1, (192, ), (1, ))
    assert_size_stride(arg412_1, (192, ), (1, ))
    assert_size_stride(arg413_1, (192, ), (1, ))
    assert_size_stride(arg414_1, (192, ), (1, ))
    assert_size_stride(arg415_1, (192, ), (1, ))
    assert_size_stride(arg416_1, (192, ), (1, ))
    assert_size_stride(arg417_1, (192, ), (1, ))
    assert_size_stride(arg418_1, (192, ), (1, ))
    assert_size_stride(arg419_1, (192, ), (1, ))
    assert_size_stride(arg420_1, (192, ), (1, ))
    assert_size_stride(arg421_1, (192, ), (1, ))
    assert_size_stride(arg422_1, (192, ), (1, ))
    assert_size_stride(arg423_1, (192, ), (1, ))
    assert_size_stride(arg424_1, (192, ), (1, ))
    assert_size_stride(arg425_1, (192, ), (1, ))
    assert_size_stride(arg426_1, (320, ), (1, ))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (192, ), (1, ))
    assert_size_stride(arg429_1, (192, ), (1, ))
    assert_size_stride(arg430_1, (192, ), (1, ))
    assert_size_stride(arg431_1, (192, ), (1, ))
    assert_size_stride(arg432_1, (192, ), (1, ))
    assert_size_stride(arg433_1, (192, ), (1, ))
    assert_size_stride(arg434_1, (192, ), (1, ))
    assert_size_stride(arg435_1, (192, ), (1, ))
    assert_size_stride(arg436_1, (320, ), (1, ))
    assert_size_stride(arg437_1, (320, ), (1, ))
    assert_size_stride(arg438_1, (384, ), (1, ))
    assert_size_stride(arg439_1, (384, ), (1, ))
    assert_size_stride(arg440_1, (384, ), (1, ))
    assert_size_stride(arg441_1, (384, ), (1, ))
    assert_size_stride(arg442_1, (384, ), (1, ))
    assert_size_stride(arg443_1, (384, ), (1, ))
    assert_size_stride(arg444_1, (448, ), (1, ))
    assert_size_stride(arg445_1, (448, ), (1, ))
    assert_size_stride(arg446_1, (384, ), (1, ))
    assert_size_stride(arg447_1, (384, ), (1, ))
    assert_size_stride(arg448_1, (384, ), (1, ))
    assert_size_stride(arg449_1, (384, ), (1, ))
    assert_size_stride(arg450_1, (384, ), (1, ))
    assert_size_stride(arg451_1, (384, ), (1, ))
    assert_size_stride(arg452_1, (192, ), (1, ))
    assert_size_stride(arg453_1, (192, ), (1, ))
    assert_size_stride(arg454_1, (320, ), (1, ))
    assert_size_stride(arg455_1, (320, ), (1, ))
    assert_size_stride(arg456_1, (384, ), (1, ))
    assert_size_stride(arg457_1, (384, ), (1, ))
    assert_size_stride(arg458_1, (384, ), (1, ))
    assert_size_stride(arg459_1, (384, ), (1, ))
    assert_size_stride(arg460_1, (384, ), (1, ))
    assert_size_stride(arg461_1, (384, ), (1, ))
    assert_size_stride(arg462_1, (448, ), (1, ))
    assert_size_stride(arg463_1, (448, ), (1, ))
    assert_size_stride(arg464_1, (384, ), (1, ))
    assert_size_stride(arg465_1, (384, ), (1, ))
    assert_size_stride(arg466_1, (384, ), (1, ))
    assert_size_stride(arg467_1, (384, ), (1, ))
    assert_size_stride(arg468_1, (384, ), (1, ))
    assert_size_stride(arg469_1, (384, ), (1, ))
    assert_size_stride(arg470_1, (192, ), (1, ))
    assert_size_stride(arg471_1, (192, ), (1, ))
    assert_size_stride(arg472_1, (8, 3, 299, 299), (268203, 89401, 299, 1))
    buf0 = empty_strided((8, 3, 299, 299), (268203, 1, 897, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg472_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg188_1
    del arg472_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 149, 149), (710432, 1, 4768, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg189_1
    del arg1_1
    del arg284_1
    del arg285_1
    # Source Nodes: [x_1, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 32, 147, 147), (691488, 1, 4704, 32))
    del buf3
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg190_1
    del arg286_1
    del arg287_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 64, 147, 147), (1382976, 1, 9408, 64))
    del buf6
    del buf7
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(c_void_p(buf9.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf10.data_ptr()))
    del arg288_1
    del arg289_1
    del arg4_1
    del arg5_1
    del buf9
    # Source Nodes: [x_19], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 80, 73, 73), (426320, 1, 5840, 80))
    del arg191_1
    del buf10
    buf12 = buf11; del buf11  # reuse
    buf13 = empty_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg290_1.data_ptr()), c_void_p(arg291_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg192_1
    del arg290_1
    del arg291_1
    del arg6_1
    del arg7_1
    # Source Nodes: [x_20, x_24, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 192, 71, 71), (967872, 1, 13632, 192))
    del buf12
    del buf13
    buf15 = buf14; del buf14  # reuse
    buf16 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_5(c_void_p(buf15.data_ptr()), c_void_p(arg292_1.data_ptr()), c_void_p(arg293_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg292_1
    del arg293_1
    del arg8_1
    del arg9_1
    del buf15
    # Source Nodes: [x_32], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(buf16, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf17, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg193_1
    # Source Nodes: [x_37], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf16, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (8, 48, 35, 35), (58800, 1, 1680, 48))
    del arg194_1
    buf19 = buf18; del buf18  # reuse
    buf20 = empty_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_6(c_void_p(buf19.data_ptr()), c_void_p(arg296_1.data_ptr()), c_void_p(arg297_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg12_1
    del arg13_1
    del arg195_1
    del arg296_1
    del arg297_1
    # Source Nodes: [branch5x5, x_38, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del buf19
    # Source Nodes: [x_47], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf16, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf22, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg196_1
    buf23 = buf22; del buf22  # reuse
    buf24 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7(c_void_p(buf23.data_ptr()), c_void_p(arg300_1.data_ptr()), c_void_p(arg301_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(buf24.data_ptr()))
    del arg16_1
    del arg17_1
    del arg197_1
    del arg300_1
    del arg301_1
    # Source Nodes: [branch3x3dbl, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf25, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf23
    buf26 = buf25; del buf25  # reuse
    buf27 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8(c_void_p(buf26.data_ptr()), c_void_p(arg302_1.data_ptr()), c_void_p(arg303_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(buf27.data_ptr()))
    del arg18_1
    del arg198_1
    del arg19_1
    del arg302_1
    del arg303_1
    # Source Nodes: [branch3x3dbl_1, x_53, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf26
    buf29 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_9(c_void_p(buf16.data_ptr()), c_void_p(buf29.data_ptr()))
    del buf16
    # Source Nodes: [x_62], Original ATen: [aten.convolution]
    buf30 = extern_kernels.convolution(buf29, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf30, (8, 32, 35, 35), (39200, 1, 1120, 32))
    del arg199_1
    del buf29
    buf31 = empty_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cpu', dtype=torch.float32)
    cpp_fused_cat_10(c_void_p(buf17.data_ptr()), c_void_p(arg294_1.data_ptr()), c_void_p(arg295_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg298_1.data_ptr()), c_void_p(arg299_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(arg304_1.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg10_1
    del arg11_1
    del arg14_1
    del arg15_1
    del arg20_1
    del arg21_1
    del arg22_1
    del arg23_1
    del arg294_1
    del arg295_1
    del arg298_1
    del arg299_1
    del arg304_1
    del arg305_1
    del arg306_1
    del arg307_1
    del buf17
    del buf21
    del buf28
    del buf30
    # Source Nodes: [x_68], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg200_1
    # Source Nodes: [x_73], Original ATen: [aten.convolution]
    buf33 = extern_kernels.convolution(buf31, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf33, (8, 48, 35, 35), (58800, 1, 1680, 48))
    del arg201_1
    buf34 = buf33; del buf33  # reuse
    buf35 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11(c_void_p(buf34.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(buf35.data_ptr()))
    del arg202_1
    del arg26_1
    del arg27_1
    del arg310_1
    del arg311_1
    # Source Nodes: [branch5x5_2, x_74, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf36, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del buf34
    # Source Nodes: [x_83], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf31, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg203_1
    buf38 = buf37; del buf37  # reuse
    buf39 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_12(c_void_p(buf38.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(buf39.data_ptr()))
    del arg204_1
    del arg30_1
    del arg314_1
    del arg315_1
    del arg31_1
    # Source Nodes: [branch3x3dbl_3, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf38
    buf41 = buf40; del buf40  # reuse
    buf42 = buf27; del buf27  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_13(c_void_p(buf41.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(buf42.data_ptr()))
    del arg205_1
    del arg316_1
    del arg317_1
    del arg32_1
    del arg33_1
    # Source Nodes: [branch3x3dbl_4, x_89, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf43, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf41
    buf44 = empty_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_14(c_void_p(buf31.data_ptr()), c_void_p(buf44.data_ptr()))
    del buf31
    # Source Nodes: [x_98], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg206_1
    del buf44
    buf46 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cpu', dtype=torch.float32)
    cpp_fused_cat_15(c_void_p(buf32.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg24_1
    del arg25_1
    del arg28_1
    del arg29_1
    del arg308_1
    del arg309_1
    del arg312_1
    del arg313_1
    del arg318_1
    del arg319_1
    del arg320_1
    del arg321_1
    del arg34_1
    del arg35_1
    del arg36_1
    del arg37_1
    del buf32
    del buf36
    del buf43
    del buf45
    # Source Nodes: [x_104], Original ATen: [aten.convolution]
    buf47 = extern_kernels.convolution(buf46, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg207_1
    # Source Nodes: [x_109], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(buf46, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (8, 48, 35, 35), (58800, 1, 1680, 48))
    del arg208_1
    buf49 = buf48; del buf48  # reuse
    buf50 = buf35; del buf35  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_16(c_void_p(buf49.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(buf50.data_ptr()))
    del arg209_1
    del arg324_1
    del arg325_1
    del arg40_1
    del arg41_1
    # Source Nodes: [branch5x5_4, x_110, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf51 = extern_kernels.convolution(buf49, buf50, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del buf49
    del buf50
    # Source Nodes: [x_119], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf46, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg210_1
    buf53 = buf52; del buf52  # reuse
    buf54 = buf39; del buf39  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf53.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg211_1
    del arg328_1
    del arg329_1
    del arg44_1
    del arg45_1
    # Source Nodes: [branch3x3dbl_6, x_120, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf55, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf53
    buf56 = buf55; del buf55  # reuse
    buf57 = buf42; del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_18(c_void_p(buf56.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg212_1
    del arg330_1
    del arg331_1
    del arg46_1
    del arg47_1
    # Source Nodes: [branch3x3dbl_7, x_125, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf58, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf56
    buf59 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_19(c_void_p(buf46.data_ptr()), c_void_p(buf59.data_ptr()))
    del buf46
    # Source Nodes: [x_134], Original ATen: [aten.convolution]
    buf60 = extern_kernels.convolution(buf59, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf60, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg213_1
    buf61 = buf59; del buf59  # reuse
    buf74 = empty((8, 768, 17, 17), device='cpu', dtype=torch.float32)
    buf62 = reinterpret_tensor(buf74, (8, 288, 17, 17), (221952, 289, 17, 1), 138720)  # alias
    buf63 = empty_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cpu', dtype=torch.float32)
    cpp_fused_cat_convolution_max_pool2d_with_indices_20(c_void_p(buf47.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf58.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg214_1
    del arg322_1
    del arg323_1
    del arg326_1
    del arg327_1
    del arg332_1
    del arg333_1
    del arg334_1
    del arg335_1
    del arg38_1
    del arg39_1
    del arg42_1
    del arg43_1
    del arg48_1
    del arg49_1
    del arg50_1
    del arg51_1
    del buf47
    del buf51
    del buf58
    del buf60
    # Source Nodes: [x_140], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf61, buf63, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 384, 17, 17), (110976, 1, 6528, 384))
    del buf63
    # Source Nodes: [x_145], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf61, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (8, 64, 35, 35), (78400, 1, 2240, 64))
    del arg215_1
    del buf61
    buf66 = buf65; del buf65  # reuse
    buf67 = buf54; del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_21(c_void_p(buf66.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(buf67.data_ptr()))
    del arg216_1
    del arg338_1
    del arg339_1
    del arg54_1
    del arg55_1
    # Source Nodes: [branch3x3dbl_9, x_146, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf68 = extern_kernels.convolution(buf66, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 96, 35, 35), (117600, 1, 3360, 96))
    del buf66
    del buf67
    buf69 = buf68; del buf68  # reuse
    buf70 = buf57; del buf57  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_22(c_void_p(buf69.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(buf70.data_ptr()))
    del arg217_1
    del arg340_1
    del arg341_1
    del arg56_1
    del arg57_1
    # Source Nodes: [branch3x3dbl_10, x_151, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf71 = extern_kernels.convolution(buf69, buf70, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (8, 96, 17, 17), (27744, 1, 1632, 96))
    del buf69
    del buf70
    buf72 = reinterpret_tensor(buf74, (8, 384, 17, 17), (221952, 289, 17, 1), 0)  # alias
    buf73 = reinterpret_tensor(buf74, (8, 96, 17, 17), (221952, 289, 17, 1), 110976)  # alias
    buf75 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    buf77 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23(c_void_p(buf64.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg336_1
    del arg337_1
    del arg342_1
    del arg343_1
    del arg52_1
    del arg53_1
    del arg58_1
    del arg59_1
    del buf62
    del buf64
    del buf71
    del buf72
    del buf73
    # Source Nodes: [x_161], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf76, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg218_1
    del buf75
    # Source Nodes: [x_166], Original ATen: [aten.convolution]
    buf78 = extern_kernels.convolution(buf77, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del arg219_1
    del buf77
    buf79 = buf78; del buf78  # reuse
    buf80 = empty_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_24(c_void_p(buf79.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(buf80.data_ptr()))
    del arg220_1
    del arg346_1
    del arg347_1
    del arg62_1
    del arg63_1
    # Source Nodes: [branch7x7, x_167, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del buf79
    buf82 = buf81; del buf81  # reuse
    buf83 = empty_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_25(c_void_p(buf82.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg221_1
    del arg348_1
    del arg349_1
    del arg64_1
    del arg65_1
    # Source Nodes: [branch7x7_1, x_172, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf82
    # Source Nodes: [x_181], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del arg222_1
    buf87 = buf86; del buf86  # reuse
    buf88 = reinterpret_tensor(buf80, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_26(c_void_p(buf87.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg223_1
    del arg352_1
    del arg353_1
    del arg68_1
    del arg69_1
    # Source Nodes: [branch7x7dbl, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del buf87
    buf90 = buf89; del buf89  # reuse
    buf91 = reinterpret_tensor(buf88, (128, 128, 1, 7), (896, 1, 896, 128), 0); del buf88  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_27(c_void_p(buf90.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(buf91.data_ptr()))
    del arg224_1
    del arg354_1
    del arg355_1
    del arg70_1
    del arg71_1
    # Source Nodes: [branch7x7dbl_1, x_187, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del buf90
    buf93 = buf92; del buf92  # reuse
    buf94 = reinterpret_tensor(buf91, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_28(c_void_p(buf93.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg225_1
    del arg356_1
    del arg357_1
    del arg72_1
    del arg73_1
    # Source Nodes: [branch7x7dbl_2, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf95 = extern_kernels.convolution(buf93, buf94, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf95, (8, 128, 17, 17), (36992, 1, 2176, 128))
    del buf93
    del buf94
    buf96 = buf95; del buf95  # reuse
    buf97 = reinterpret_tensor(buf83, (192, 128, 1, 7), (896, 1, 896, 128), 0); del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_29(c_void_p(buf96.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(buf97.data_ptr()))
    del arg226_1
    del arg358_1
    del arg359_1
    del arg74_1
    del arg75_1
    # Source Nodes: [branch7x7dbl_3, x_197, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf98, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf96
    del buf97
    buf99 = buf85; del buf85  # reuse
    cpp_fused_avg_pool2d_30(c_void_p(buf74.data_ptr()), c_void_p(buf99.data_ptr()))
    # Source Nodes: [x_206], Original ATen: [aten.convolution]
    buf100 = extern_kernels.convolution(buf99, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg227_1
    buf101 = buf99; del buf99  # reuse
    cpp_fused_cat_31(c_void_p(buf76.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf101.data_ptr()))
    del arg344_1
    del arg345_1
    del arg350_1
    del arg351_1
    del arg360_1
    del arg361_1
    del arg362_1
    del arg363_1
    del arg60_1
    del arg61_1
    del arg66_1
    del arg67_1
    del arg76_1
    del arg77_1
    del arg78_1
    del arg79_1
    del buf100
    del buf76
    del buf84
    del buf98
    # Source Nodes: [x_212], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg228_1
    # Source Nodes: [x_217], Original ATen: [aten.convolution]
    buf103 = extern_kernels.convolution(buf101, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf103, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del arg229_1
    buf104 = buf103; del buf103  # reuse
    buf105 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_32(c_void_p(buf104.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(buf105.data_ptr()))
    del arg230_1
    del arg366_1
    del arg367_1
    del arg82_1
    del arg83_1
    # Source Nodes: [branch7x7_3, x_218, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf104
    buf107 = buf106; del buf106  # reuse
    buf108 = empty_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_33(c_void_p(buf107.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg231_1
    del arg368_1
    del arg369_1
    del arg84_1
    del arg85_1
    # Source Nodes: [branch7x7_4, x_223, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf109 = extern_kernels.convolution(buf107, buf108, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf109, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf107
    # Source Nodes: [x_232], Original ATen: [aten.convolution]
    buf110 = extern_kernels.convolution(buf101, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del arg232_1
    buf111 = buf110; del buf110  # reuse
    buf112 = reinterpret_tensor(buf105, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf105  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_34(c_void_p(buf111.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(buf112.data_ptr()))
    del arg233_1
    del arg372_1
    del arg373_1
    del arg88_1
    del arg89_1
    # Source Nodes: [branch7x7dbl_5, x_233, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf113, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf111
    buf114 = buf113; del buf113  # reuse
    buf115 = reinterpret_tensor(buf112, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf112  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_35(c_void_p(buf114.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(buf115.data_ptr()))
    del arg234_1
    del arg374_1
    del arg375_1
    del arg90_1
    del arg91_1
    # Source Nodes: [branch7x7dbl_6, x_238, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf116, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf114
    buf117 = buf116; del buf116  # reuse
    buf118 = reinterpret_tensor(buf115, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf115  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_36(c_void_p(buf117.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg235_1
    del arg376_1
    del arg377_1
    del arg92_1
    del arg93_1
    # Source Nodes: [branch7x7dbl_7, x_243, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf119, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf117
    buf120 = buf119; del buf119  # reuse
    buf121 = reinterpret_tensor(buf108, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_37(c_void_p(buf120.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(buf121.data_ptr()))
    del arg236_1
    del arg378_1
    del arg379_1
    del arg94_1
    del arg95_1
    # Source Nodes: [branch7x7dbl_8, x_248, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf122, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf120
    buf123 = reinterpret_tensor(buf74, (8, 768, 17, 17), (221952, 1, 13056, 768), 0); del buf74  # reuse
    cpp_fused_avg_pool2d_38(c_void_p(buf101.data_ptr()), c_void_p(buf123.data_ptr()))
    # Source Nodes: [x_257], Original ATen: [aten.convolution]
    buf124 = extern_kernels.convolution(buf123, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg237_1
    buf125 = buf123; del buf123  # reuse
    cpp_fused_cat_39(c_void_p(buf102.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf122.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf125.data_ptr()))
    del arg364_1
    del arg365_1
    del arg370_1
    del arg371_1
    del arg380_1
    del arg381_1
    del arg382_1
    del arg383_1
    del arg80_1
    del arg81_1
    del arg86_1
    del arg87_1
    del arg96_1
    del arg97_1
    del arg98_1
    del arg99_1
    del buf102
    del buf109
    del buf122
    del buf124
    # Source Nodes: [x_263], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(buf125, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg238_1
    # Source Nodes: [x_268], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(buf125, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del arg239_1
    buf128 = buf127; del buf127  # reuse
    buf129 = reinterpret_tensor(buf118, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_40(c_void_p(buf128.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(buf129.data_ptr()))
    del arg102_1
    del arg103_1
    del arg240_1
    del arg386_1
    del arg387_1
    # Source Nodes: [branch7x7_6, x_269, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf130 = extern_kernels.convolution(buf128, buf129, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf130, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf128
    buf131 = buf130; del buf130  # reuse
    buf132 = reinterpret_tensor(buf121, (192, 160, 7, 1), (1120, 1, 160, 160), 0); del buf121  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_41(c_void_p(buf131.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(buf132.data_ptr()))
    del arg104_1
    del arg105_1
    del arg241_1
    del arg388_1
    del arg389_1
    # Source Nodes: [branch7x7_7, x_274, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf133 = extern_kernels.convolution(buf131, buf132, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf133, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf131
    # Source Nodes: [x_283], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf125, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf134, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del arg242_1
    buf135 = buf134; del buf134  # reuse
    buf136 = reinterpret_tensor(buf129, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf129  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_42(c_void_p(buf135.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(buf136.data_ptr()))
    del arg108_1
    del arg109_1
    del arg243_1
    del arg392_1
    del arg393_1
    # Source Nodes: [branch7x7dbl_10, x_284, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf137, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf135
    buf138 = buf137; del buf137  # reuse
    buf139 = reinterpret_tensor(buf136, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf136  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_43(c_void_p(buf138.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(buf139.data_ptr()))
    del arg110_1
    del arg111_1
    del arg244_1
    del arg394_1
    del arg395_1
    # Source Nodes: [branch7x7dbl_11, x_289, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf140, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf138
    buf141 = buf140; del buf140  # reuse
    buf142 = reinterpret_tensor(buf139, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_44(c_void_p(buf141.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(buf142.data_ptr()))
    del arg112_1
    del arg113_1
    del arg245_1
    del arg396_1
    del arg397_1
    # Source Nodes: [branch7x7dbl_12, x_294, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf143, (8, 160, 17, 17), (46240, 1, 2720, 160))
    del buf141
    del buf142
    buf144 = buf143; del buf143  # reuse
    buf145 = reinterpret_tensor(buf132, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf132  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_45(c_void_p(buf144.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(buf145.data_ptr()))
    del arg114_1
    del arg115_1
    del arg246_1
    del arg398_1
    del arg399_1
    # Source Nodes: [branch7x7dbl_13, x_299, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf146 = extern_kernels.convolution(buf144, buf145, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf144
    del buf145
    buf147 = buf101; del buf101  # reuse
    cpp_fused_avg_pool2d_46(c_void_p(buf125.data_ptr()), c_void_p(buf147.data_ptr()))
    # Source Nodes: [x_308], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf147, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf148, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg247_1
    buf149 = buf147; del buf147  # reuse
    cpp_fused_cat_47(c_void_p(buf126.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf149.data_ptr()))
    del arg100_1
    del arg101_1
    del arg106_1
    del arg107_1
    del arg116_1
    del arg117_1
    del arg118_1
    del arg119_1
    del arg384_1
    del arg385_1
    del arg390_1
    del arg391_1
    del arg400_1
    del arg401_1
    del arg402_1
    del arg403_1
    del buf126
    del buf133
    del buf146
    del buf148
    # Source Nodes: [x_314], Original ATen: [aten.convolution]
    buf150 = extern_kernels.convolution(buf149, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf150, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg248_1
    # Source Nodes: [x_319], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(buf149, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg249_1
    buf152 = buf151; del buf151  # reuse
    buf153 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_48(c_void_p(buf152.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(buf153.data_ptr()))
    del arg122_1
    del arg123_1
    del arg250_1
    del arg406_1
    del arg407_1
    # Source Nodes: [branch7x7_9, x_320, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf154, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf152
    buf155 = buf154; del buf154  # reuse
    buf156 = reinterpret_tensor(buf153, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf153  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_49(c_void_p(buf155.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(buf156.data_ptr()))
    del arg124_1
    del arg125_1
    del arg251_1
    del arg408_1
    del arg409_1
    # Source Nodes: [branch7x7_10, x_325, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf157, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf155
    # Source Nodes: [x_334], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf149, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf158, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg252_1
    buf159 = buf158; del buf158  # reuse
    buf160 = buf156; del buf156  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_50(c_void_p(buf159.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(buf160.data_ptr()))
    del arg128_1
    del arg129_1
    del arg253_1
    del arg412_1
    del arg413_1
    # Source Nodes: [branch7x7dbl_15, x_335, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf161, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf159
    buf162 = buf161; del buf161  # reuse
    buf163 = reinterpret_tensor(buf160, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf160  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_51(c_void_p(buf162.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg130_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg130_1
    del arg131_1
    del arg254_1
    del arg414_1
    del arg415_1
    # Source Nodes: [branch7x7dbl_16, x_340, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf164, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf162
    buf165 = buf164; del buf164  # reuse
    buf166 = reinterpret_tensor(buf163, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_52(c_void_p(buf165.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg132_1.data_ptr()), c_void_p(arg133_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(buf166.data_ptr()))
    del arg132_1
    del arg133_1
    del arg255_1
    del arg416_1
    del arg417_1
    # Source Nodes: [branch7x7dbl_17, x_345, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf167 = extern_kernels.convolution(buf165, buf166, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf167, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf165
    buf168 = buf167; del buf167  # reuse
    buf169 = reinterpret_tensor(buf166, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf166  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_53(c_void_p(buf168.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(arg135_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(buf169.data_ptr()))
    del arg134_1
    del arg135_1
    del arg256_1
    del arg418_1
    del arg419_1
    # Source Nodes: [branch7x7dbl_18, x_350, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf170 = extern_kernels.convolution(buf168, buf169, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf168
    buf171 = buf125; del buf125  # reuse
    cpp_fused_avg_pool2d_54(c_void_p(buf149.data_ptr()), c_void_p(buf171.data_ptr()))
    del buf149
    # Source Nodes: [x_359], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg257_1
    buf173 = buf171; del buf171  # reuse
    buf191 = empty((8, 1280, 8, 8), device='cpu', dtype=torch.float32)
    buf174 = reinterpret_tensor(buf191, (8, 768, 8, 8), (81920, 64, 8, 1), 32768)  # alias
    cpp_fused_cat_max_pool2d_with_indices_55(c_void_p(buf150.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg421_1.data_ptr()), c_void_p(arg136_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg422_1.data_ptr()), c_void_p(arg423_1.data_ptr()), c_void_p(arg138_1.data_ptr()), c_void_p(arg139_1.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del arg120_1
    del arg121_1
    del arg126_1
    del arg127_1
    del arg136_1
    del arg137_1
    del arg138_1
    del arg139_1
    del arg404_1
    del arg405_1
    del arg410_1
    del arg411_1
    del arg420_1
    del arg421_1
    del arg422_1
    del arg423_1
    del buf150
    del buf157
    del buf170
    del buf172
    # Source Nodes: [x_366], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(buf173, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg258_1
    buf176 = buf175; del buf175  # reuse
    buf177 = empty_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_56(c_void_p(buf176.data_ptr()), c_void_p(arg424_1.data_ptr()), c_void_p(arg425_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(arg141_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(buf177.data_ptr()))
    del arg140_1
    del arg141_1
    del arg259_1
    del arg424_1
    del arg425_1
    # Source Nodes: [branch3x3_1, x_367, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf178 = extern_kernels.convolution(buf176, buf177, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf178, (8, 320, 8, 8), (20480, 1, 2560, 320))
    del buf176
    del buf177
    # Source Nodes: [x_376], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf173, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf179, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del arg260_1
    del buf173
    buf180 = buf179; del buf179  # reuse
    buf181 = buf169; del buf169  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_57(c_void_p(buf180.data_ptr()), c_void_p(arg428_1.data_ptr()), c_void_p(arg429_1.data_ptr()), c_void_p(arg144_1.data_ptr()), c_void_p(arg145_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(buf181.data_ptr()))
    del arg144_1
    del arg145_1
    del arg261_1
    del arg428_1
    del arg429_1
    # Source Nodes: [branch7x7x3, x_377, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf182, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf180
    buf183 = buf182; del buf182  # reuse
    buf184 = reinterpret_tensor(buf181, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf181  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_58(c_void_p(buf183.data_ptr()), c_void_p(arg430_1.data_ptr()), c_void_p(arg431_1.data_ptr()), c_void_p(arg146_1.data_ptr()), c_void_p(arg147_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(buf184.data_ptr()))
    del arg146_1
    del arg147_1
    del arg262_1
    del arg430_1
    del arg431_1
    # Source Nodes: [branch7x7x3_1, x_382, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf185, (8, 192, 17, 17), (55488, 1, 3264, 192))
    del buf183
    del buf184
    buf186 = buf185; del buf185  # reuse
    buf187 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_59(c_void_p(buf186.data_ptr()), c_void_p(arg432_1.data_ptr()), c_void_p(arg433_1.data_ptr()), c_void_p(arg148_1.data_ptr()), c_void_p(arg149_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg148_1
    del arg149_1
    del arg263_1
    del arg432_1
    del arg433_1
    # Source Nodes: [branch7x7x3_2, x_387, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf188 = extern_kernels.convolution(buf186, buf187, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf188, (8, 192, 8, 8), (12288, 1, 1536, 192))
    del buf186
    del buf187
    buf189 = reinterpret_tensor(buf191, (8, 320, 8, 8), (81920, 64, 8, 1), 0)  # alias
    buf190 = reinterpret_tensor(buf191, (8, 192, 8, 8), (81920, 64, 8, 1), 20480)  # alias
    buf192 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    buf194 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    buf202 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_60(c_void_p(buf178.data_ptr()), c_void_p(arg426_1.data_ptr()), c_void_p(arg427_1.data_ptr()), c_void_p(arg142_1.data_ptr()), c_void_p(arg143_1.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(arg434_1.data_ptr()), c_void_p(arg435_1.data_ptr()), c_void_p(arg150_1.data_ptr()), c_void_p(arg151_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf202.data_ptr()))
    del arg142_1
    del arg143_1
    del arg150_1
    del arg151_1
    del arg426_1
    del arg427_1
    del arg434_1
    del arg435_1
    del buf174
    del buf178
    del buf188
    del buf189
    del buf190
    # Source Nodes: [x_397], Original ATen: [aten.convolution]
    buf193 = extern_kernels.convolution(buf192, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf193, (8, 320, 8, 8), (20480, 1, 2560, 320))
    del arg264_1
    del buf192
    # Source Nodes: [x_402], Original ATen: [aten.convolution]
    buf195 = extern_kernels.convolution(buf194, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf195, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del arg265_1
    del buf194
    buf196 = buf195; del buf195  # reuse
    buf197 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_61(c_void_p(buf196.data_ptr()), c_void_p(arg438_1.data_ptr()), c_void_p(arg439_1.data_ptr()), c_void_p(arg154_1.data_ptr()), c_void_p(arg155_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(buf197.data_ptr()))
    del arg154_1
    del arg155_1
    del arg266_1
    del arg438_1
    del arg439_1
    # Source Nodes: [x_407], Original ATen: [aten.convolution]
    buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf198, (8, 384, 8, 8), (24576, 1, 3072, 384))
    buf199 = reinterpret_tensor(buf197, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf197  # reuse
    cpp_fused_convolution_62(c_void_p(arg267_1.data_ptr()), c_void_p(buf199.data_ptr()))
    del arg267_1
    # Source Nodes: [x_412], Original ATen: [aten.convolution]
    buf200 = extern_kernels.convolution(buf196, buf199, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf200, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf196
    buf217 = empty((8, 2048, 8, 8), device='cpu', dtype=torch.float32)
    buf201 = reinterpret_tensor(buf217, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
    cpp_fused_cat_63(c_void_p(buf198.data_ptr()), c_void_p(arg440_1.data_ptr()), c_void_p(arg441_1.data_ptr()), c_void_p(arg156_1.data_ptr()), c_void_p(arg157_1.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(arg442_1.data_ptr()), c_void_p(arg443_1.data_ptr()), c_void_p(arg158_1.data_ptr()), c_void_p(arg159_1.data_ptr()), c_void_p(buf201.data_ptr()))
    del arg156_1
    del arg157_1
    del arg158_1
    del arg159_1
    del arg440_1
    del arg441_1
    del arg442_1
    del arg443_1
    del buf198
    del buf200
    # Source Nodes: [x_417], Original ATen: [aten.convolution]
    buf203 = extern_kernels.convolution(buf202, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf203, (8, 448, 8, 8), (28672, 1, 3584, 448))
    del arg268_1
    buf204 = buf203; del buf203  # reuse
    buf205 = empty_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_64(c_void_p(buf204.data_ptr()), c_void_p(arg444_1.data_ptr()), c_void_p(arg445_1.data_ptr()), c_void_p(arg160_1.data_ptr()), c_void_p(arg161_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(buf205.data_ptr()))
    del arg160_1
    del arg161_1
    del arg269_1
    del arg444_1
    del arg445_1
    # Source Nodes: [branch3x3dbl_12, x_418, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf206 = extern_kernels.convolution(buf204, buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf206, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf204
    buf207 = buf206; del buf206  # reuse
    buf208 = reinterpret_tensor(buf199, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf199  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_65(c_void_p(buf207.data_ptr()), c_void_p(arg446_1.data_ptr()), c_void_p(arg447_1.data_ptr()), c_void_p(arg162_1.data_ptr()), c_void_p(arg163_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(buf208.data_ptr()))
    del arg162_1
    del arg163_1
    del arg270_1
    del arg446_1
    del arg447_1
    # Source Nodes: [x_427], Original ATen: [aten.convolution]
    buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf209, (8, 384, 8, 8), (24576, 1, 3072, 384))
    buf210 = reinterpret_tensor(buf208, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf208  # reuse
    cpp_fused_convolution_66(c_void_p(arg271_1.data_ptr()), c_void_p(buf210.data_ptr()))
    del arg271_1
    # Source Nodes: [x_432], Original ATen: [aten.convolution]
    buf211 = extern_kernels.convolution(buf207, buf210, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf211, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf207
    buf212 = reinterpret_tensor(buf217, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
    buf213 = buf202; del buf202  # reuse
    cpp_fused_avg_pool2d_cat_67(c_void_p(buf209.data_ptr()), c_void_p(arg448_1.data_ptr()), c_void_p(arg449_1.data_ptr()), c_void_p(arg164_1.data_ptr()), c_void_p(arg165_1.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(arg450_1.data_ptr()), c_void_p(arg451_1.data_ptr()), c_void_p(arg166_1.data_ptr()), c_void_p(arg167_1.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del arg164_1
    del arg165_1
    del arg166_1
    del arg167_1
    del arg448_1
    del arg449_1
    del arg450_1
    del arg451_1
    del buf191
    del buf209
    del buf211
    # Source Nodes: [x_437], Original ATen: [aten.convolution]
    buf214 = extern_kernels.convolution(buf213, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf214, (8, 192, 8, 8), (12288, 1, 1536, 192))
    del arg272_1
    del buf213
    buf215 = reinterpret_tensor(buf217, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
    buf216 = reinterpret_tensor(buf217, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
    buf218 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf220 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    buf228 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_68(c_void_p(buf193.data_ptr()), c_void_p(arg436_1.data_ptr()), c_void_p(arg437_1.data_ptr()), c_void_p(arg152_1.data_ptr()), c_void_p(arg153_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(arg452_1.data_ptr()), c_void_p(arg453_1.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf228.data_ptr()))
    del arg152_1
    del arg153_1
    del arg168_1
    del arg169_1
    del arg436_1
    del arg437_1
    del arg452_1
    del arg453_1
    del buf193
    del buf201
    del buf212
    del buf214
    del buf215
    del buf216
    # Source Nodes: [x_443], Original ATen: [aten.convolution]
    buf219 = extern_kernels.convolution(buf218, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf219, (8, 320, 8, 8), (20480, 1, 2560, 320))
    del arg273_1
    del buf218
    # Source Nodes: [x_448], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf221, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del arg274_1
    buf222 = buf221; del buf221  # reuse
    buf223 = reinterpret_tensor(buf210, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_69(c_void_p(buf222.data_ptr()), c_void_p(arg456_1.data_ptr()), c_void_p(arg457_1.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg172_1
    del arg173_1
    del arg275_1
    del arg456_1
    del arg457_1
    # Source Nodes: [x_453], Original ATen: [aten.convolution]
    buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf224, (8, 384, 8, 8), (24576, 1, 3072, 384))
    buf225 = reinterpret_tensor(buf223, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf223  # reuse
    cpp_fused_convolution_70(c_void_p(arg276_1.data_ptr()), c_void_p(buf225.data_ptr()))
    del arg276_1
    # Source Nodes: [x_458], Original ATen: [aten.convolution]
    buf226 = extern_kernels.convolution(buf222, buf225, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf226, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf222
    buf243 = reinterpret_tensor(buf220, (8, 2048, 8, 8), (131072, 64, 8, 1), 0); del buf220  # reuse
    buf227 = reinterpret_tensor(buf243, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
    cpp_fused_cat_71(c_void_p(buf224.data_ptr()), c_void_p(arg458_1.data_ptr()), c_void_p(arg459_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(buf226.data_ptr()), c_void_p(arg460_1.data_ptr()), c_void_p(arg461_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(buf227.data_ptr()))
    del arg174_1
    del arg175_1
    del arg176_1
    del arg177_1
    del arg458_1
    del arg459_1
    del arg460_1
    del arg461_1
    del buf224
    del buf226
    # Source Nodes: [x_463], Original ATen: [aten.convolution]
    buf229 = extern_kernels.convolution(buf228, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf229, (8, 448, 8, 8), (28672, 1, 3584, 448))
    del arg277_1
    buf230 = buf229; del buf229  # reuse
    buf231 = buf205; del buf205  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_72(c_void_p(buf230.data_ptr()), c_void_p(arg462_1.data_ptr()), c_void_p(arg463_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(buf231.data_ptr()))
    del arg178_1
    del arg179_1
    del arg278_1
    del arg462_1
    del arg463_1
    # Source Nodes: [branch3x3dbl_15, x_464, x_468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf232, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf230
    del buf231
    buf233 = buf232; del buf232  # reuse
    buf234 = reinterpret_tensor(buf225, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf225  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_73(c_void_p(buf233.data_ptr()), c_void_p(arg464_1.data_ptr()), c_void_p(arg465_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(buf234.data_ptr()))
    del arg180_1
    del arg181_1
    del arg279_1
    del arg464_1
    del arg465_1
    # Source Nodes: [x_473], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (8, 384, 8, 8), (24576, 1, 3072, 384))
    buf236 = reinterpret_tensor(buf234, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf234  # reuse
    cpp_fused_convolution_74(c_void_p(arg280_1.data_ptr()), c_void_p(buf236.data_ptr()))
    del arg280_1
    # Source Nodes: [x_478], Original ATen: [aten.convolution]
    buf237 = extern_kernels.convolution(buf233, buf236, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf237, (8, 384, 8, 8), (24576, 1, 3072, 384))
    del buf233
    del buf236
    buf238 = reinterpret_tensor(buf243, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
    buf239 = buf228; del buf228  # reuse
    cpp_fused_avg_pool2d_cat_75(c_void_p(buf235.data_ptr()), c_void_p(arg466_1.data_ptr()), c_void_p(arg467_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg468_1.data_ptr()), c_void_p(arg469_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del arg182_1
    del arg183_1
    del arg184_1
    del arg185_1
    del arg466_1
    del arg467_1
    del arg468_1
    del arg469_1
    del buf217
    del buf235
    del buf237
    # Source Nodes: [x_483], Original ATen: [aten.convolution]
    buf240 = extern_kernels.convolution(buf239, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf240, (8, 192, 8, 8), (12288, 1, 1536, 192))
    del arg281_1
    del buf239
    buf241 = reinterpret_tensor(buf243, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
    buf242 = reinterpret_tensor(buf243, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
    buf244 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cpu', dtype=torch.float32)
    buf245 = reinterpret_tensor(buf244, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf244  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_76(c_void_p(buf245.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(arg454_1.data_ptr()), c_void_p(arg455_1.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(arg470_1.data_ptr()), c_void_p(arg471_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf242.data_ptr()))
    del arg170_1
    del arg171_1
    del arg186_1
    del arg187_1
    del arg454_1
    del arg455_1
    del arg470_1
    del arg471_1
    del buf219
    del buf227
    del buf238
    del buf240
    del buf241
    del buf242
    del buf243
    buf246 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_496], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg283_1, reinterpret_tensor(buf245, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg282_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf246)
    del arg282_1
    del arg283_1
    return (buf246, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((48, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg422_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg423_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg424_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg425_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg426_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg428_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg429_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg430_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg431_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg432_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg433_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg434_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg435_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg436_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg437_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg438_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg439_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg440_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg441_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg442_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg443_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg444_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg445_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg446_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg447_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg448_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg449_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg450_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg451_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg452_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg453_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg454_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg455_1 = rand_strided((320, ), (1, ), device='cpu', dtype=torch.float32)
    arg456_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg457_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg458_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg459_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg460_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg461_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg462_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg463_1 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    arg464_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg465_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg466_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg467_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg468_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg469_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg470_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg471_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg472_1 = rand_strided((8, 3, 299, 299), (268203, 89401, 299, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('inception_v3', benchmark_compiled_module)
