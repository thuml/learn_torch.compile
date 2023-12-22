
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(65536L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (65536L*x1) + (196608L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (196608L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_3 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4194304L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_5 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr9[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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


cpp_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)), static_cast<long>(160L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1440L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (160L*x2) + (1440L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_24 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.cpp('''
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
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
                    auto tmp19 = tmp17 - tmp18;
                    auto tmp21 = tmp20 + tmp5;
                    auto tmp22 = tmp21.sqrt();
                    auto tmp23 = tmp22.reciprocal();
                    auto tmp24 = tmp23 * tmp10;
                    auto tmp25 = tmp19 * tmp24;
                    auto tmp27 = tmp25 * tmp26;
                    auto tmp29 = tmp27 + tmp28;
                    auto tmp30 = tmp16 + tmp29;
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_38 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_39 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_45 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_51 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
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
                    auto tmp18 = tmp16 + tmp17;
                    auto tmp19 = at::vec::clamp_min(tmp18, decltype(tmp18)(0));
                    tmp19.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2560L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2560L*x2) + (163840L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
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
                            tmp_acc0_vec = tmp_acc0_vec + tmp17;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2560L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(20480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (192, ), (1, ))
    assert_size_stride(arg11_1, (192, ), (1, ))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (192, ), (1, ))
    assert_size_stride(arg18_1, (160, ), (1, ))
    assert_size_stride(arg19_1, (160, ), (1, ))
    assert_size_stride(arg20_1, (160, ), (1, ))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (640, ), (1, ))
    assert_size_stride(arg23_1, (640, ), (1, ))
    assert_size_stride(arg24_1, (640, ), (1, ))
    assert_size_stride(arg25_1, (640, ), (1, ))
    assert_size_stride(arg26_1, (160, ), (1, ))
    assert_size_stride(arg27_1, (160, ), (1, ))
    assert_size_stride(arg28_1, (160, ), (1, ))
    assert_size_stride(arg29_1, (160, ), (1, ))
    assert_size_stride(arg30_1, (640, ), (1, ))
    assert_size_stride(arg31_1, (640, ), (1, ))
    assert_size_stride(arg32_1, (160, ), (1, ))
    assert_size_stride(arg33_1, (160, ), (1, ))
    assert_size_stride(arg34_1, (160, ), (1, ))
    assert_size_stride(arg35_1, (160, ), (1, ))
    assert_size_stride(arg36_1, (640, ), (1, ))
    assert_size_stride(arg37_1, (640, ), (1, ))
    assert_size_stride(arg38_1, (160, ), (1, ))
    assert_size_stride(arg39_1, (160, ), (1, ))
    assert_size_stride(arg40_1, (160, ), (1, ))
    assert_size_stride(arg41_1, (160, ), (1, ))
    assert_size_stride(arg42_1, (640, ), (1, ))
    assert_size_stride(arg43_1, (640, ), (1, ))
    assert_size_stride(arg44_1, (160, ), (1, ))
    assert_size_stride(arg45_1, (160, ), (1, ))
    assert_size_stride(arg46_1, (160, ), (1, ))
    assert_size_stride(arg47_1, (160, ), (1, ))
    assert_size_stride(arg48_1, (640, ), (1, ))
    assert_size_stride(arg49_1, (640, ), (1, ))
    assert_size_stride(arg50_1, (160, ), (1, ))
    assert_size_stride(arg51_1, (160, ), (1, ))
    assert_size_stride(arg52_1, (160, ), (1, ))
    assert_size_stride(arg53_1, (160, ), (1, ))
    assert_size_stride(arg54_1, (640, ), (1, ))
    assert_size_stride(arg55_1, (640, ), (1, ))
    assert_size_stride(arg56_1, (1920, ), (1, ))
    assert_size_stride(arg57_1, (1920, ), (1, ))
    assert_size_stride(arg58_1, (1920, ), (1, ))
    assert_size_stride(arg59_1, (1920, ), (1, ))
    assert_size_stride(arg60_1, (640, ), (1, ))
    assert_size_stride(arg61_1, (640, ), (1, ))
    assert_size_stride(arg62_1, (640, ), (1, ))
    assert_size_stride(arg63_1, (640, ), (1, ))
    assert_size_stride(arg64_1, (1920, ), (1, ))
    assert_size_stride(arg65_1, (1920, ), (1, ))
    assert_size_stride(arg66_1, (1920, ), (1, ))
    assert_size_stride(arg67_1, (1920, ), (1, ))
    assert_size_stride(arg68_1, (640, ), (1, ))
    assert_size_stride(arg69_1, (640, ), (1, ))
    assert_size_stride(arg70_1, (1920, ), (1, ))
    assert_size_stride(arg71_1, (1920, ), (1, ))
    assert_size_stride(arg72_1, (1920, ), (1, ))
    assert_size_stride(arg73_1, (1920, ), (1, ))
    assert_size_stride(arg74_1, (640, ), (1, ))
    assert_size_stride(arg75_1, (640, ), (1, ))
    assert_size_stride(arg76_1, (1920, ), (1, ))
    assert_size_stride(arg77_1, (1920, ), (1, ))
    assert_size_stride(arg78_1, (1920, ), (1, ))
    assert_size_stride(arg79_1, (1920, ), (1, ))
    assert_size_stride(arg80_1, (640, ), (1, ))
    assert_size_stride(arg81_1, (640, ), (1, ))
    assert_size_stride(arg82_1, (1920, ), (1, ))
    assert_size_stride(arg83_1, (1920, ), (1, ))
    assert_size_stride(arg84_1, (1920, ), (1, ))
    assert_size_stride(arg85_1, (1920, ), (1, ))
    assert_size_stride(arg86_1, (640, ), (1, ))
    assert_size_stride(arg87_1, (640, ), (1, ))
    assert_size_stride(arg88_1, (1920, ), (1, ))
    assert_size_stride(arg89_1, (1920, ), (1, ))
    assert_size_stride(arg90_1, (1920, ), (1, ))
    assert_size_stride(arg91_1, (1920, ), (1, ))
    assert_size_stride(arg92_1, (640, ), (1, ))
    assert_size_stride(arg93_1, (640, ), (1, ))
    assert_size_stride(arg94_1, (1920, ), (1, ))
    assert_size_stride(arg95_1, (1920, ), (1, ))
    assert_size_stride(arg96_1, (1920, ), (1, ))
    assert_size_stride(arg97_1, (1920, ), (1, ))
    assert_size_stride(arg98_1, (640, ), (1, ))
    assert_size_stride(arg99_1, (640, ), (1, ))
    assert_size_stride(arg100_1, (1920, ), (1, ))
    assert_size_stride(arg101_1, (1920, ), (1, ))
    assert_size_stride(arg102_1, (1920, ), (1, ))
    assert_size_stride(arg103_1, (1920, ), (1, ))
    assert_size_stride(arg104_1, (640, ), (1, ))
    assert_size_stride(arg105_1, (640, ), (1, ))
    assert_size_stride(arg106_1, (1920, ), (1, ))
    assert_size_stride(arg107_1, (1920, ), (1, ))
    assert_size_stride(arg108_1, (1920, ), (1, ))
    assert_size_stride(arg109_1, (1920, ), (1, ))
    assert_size_stride(arg110_1, (640, ), (1, ))
    assert_size_stride(arg111_1, (640, ), (1, ))
    assert_size_stride(arg112_1, (2560, ), (1, ))
    assert_size_stride(arg113_1, (2560, ), (1, ))
    assert_size_stride(arg114_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg115_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg116_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg117_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg118_1, (192, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg119_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg120_1, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg121_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg122_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg123_1, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg124_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg125_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg126_1, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg127_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg128_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg129_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg130_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg131_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg132_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg133_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg134_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg135_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg136_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg137_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg138_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg139_1, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg140_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg141_1, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg142_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg143_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg144_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg145_1, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg146_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg147_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg149_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg150_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg152_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg153_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg155_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg156_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg157_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg158_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg159_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg161_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg162_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg163_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg164_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg165_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg166_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg167_1, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg168_1, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg169_1, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg170_1, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg171_1, (1000, 2560), (2560, 1))
    assert_size_stride(arg172_1, (1000, ), (1, ))
    assert_size_stride(arg173_1, (32, ), (1, ))
    assert_size_stride(arg174_1, (32, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (128, ), (1, ))
    assert_size_stride(arg178_1, (128, ), (1, ))
    assert_size_stride(arg179_1, (128, ), (1, ))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (192, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, ), (1, ))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, ), (1, ))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (160, ), (1, ))
    assert_size_stride(arg192_1, (160, ), (1, ))
    assert_size_stride(arg193_1, (160, ), (1, ))
    assert_size_stride(arg194_1, (160, ), (1, ))
    assert_size_stride(arg195_1, (640, ), (1, ))
    assert_size_stride(arg196_1, (640, ), (1, ))
    assert_size_stride(arg197_1, (640, ), (1, ))
    assert_size_stride(arg198_1, (640, ), (1, ))
    assert_size_stride(arg199_1, (160, ), (1, ))
    assert_size_stride(arg200_1, (160, ), (1, ))
    assert_size_stride(arg201_1, (160, ), (1, ))
    assert_size_stride(arg202_1, (160, ), (1, ))
    assert_size_stride(arg203_1, (640, ), (1, ))
    assert_size_stride(arg204_1, (640, ), (1, ))
    assert_size_stride(arg205_1, (160, ), (1, ))
    assert_size_stride(arg206_1, (160, ), (1, ))
    assert_size_stride(arg207_1, (160, ), (1, ))
    assert_size_stride(arg208_1, (160, ), (1, ))
    assert_size_stride(arg209_1, (640, ), (1, ))
    assert_size_stride(arg210_1, (640, ), (1, ))
    assert_size_stride(arg211_1, (160, ), (1, ))
    assert_size_stride(arg212_1, (160, ), (1, ))
    assert_size_stride(arg213_1, (160, ), (1, ))
    assert_size_stride(arg214_1, (160, ), (1, ))
    assert_size_stride(arg215_1, (640, ), (1, ))
    assert_size_stride(arg216_1, (640, ), (1, ))
    assert_size_stride(arg217_1, (160, ), (1, ))
    assert_size_stride(arg218_1, (160, ), (1, ))
    assert_size_stride(arg219_1, (160, ), (1, ))
    assert_size_stride(arg220_1, (160, ), (1, ))
    assert_size_stride(arg221_1, (640, ), (1, ))
    assert_size_stride(arg222_1, (640, ), (1, ))
    assert_size_stride(arg223_1, (160, ), (1, ))
    assert_size_stride(arg224_1, (160, ), (1, ))
    assert_size_stride(arg225_1, (160, ), (1, ))
    assert_size_stride(arg226_1, (160, ), (1, ))
    assert_size_stride(arg227_1, (640, ), (1, ))
    assert_size_stride(arg228_1, (640, ), (1, ))
    assert_size_stride(arg229_1, (1920, ), (1, ))
    assert_size_stride(arg230_1, (1920, ), (1, ))
    assert_size_stride(arg231_1, (1920, ), (1, ))
    assert_size_stride(arg232_1, (1920, ), (1, ))
    assert_size_stride(arg233_1, (640, ), (1, ))
    assert_size_stride(arg234_1, (640, ), (1, ))
    assert_size_stride(arg235_1, (640, ), (1, ))
    assert_size_stride(arg236_1, (640, ), (1, ))
    assert_size_stride(arg237_1, (1920, ), (1, ))
    assert_size_stride(arg238_1, (1920, ), (1, ))
    assert_size_stride(arg239_1, (1920, ), (1, ))
    assert_size_stride(arg240_1, (1920, ), (1, ))
    assert_size_stride(arg241_1, (640, ), (1, ))
    assert_size_stride(arg242_1, (640, ), (1, ))
    assert_size_stride(arg243_1, (1920, ), (1, ))
    assert_size_stride(arg244_1, (1920, ), (1, ))
    assert_size_stride(arg245_1, (1920, ), (1, ))
    assert_size_stride(arg246_1, (1920, ), (1, ))
    assert_size_stride(arg247_1, (640, ), (1, ))
    assert_size_stride(arg248_1, (640, ), (1, ))
    assert_size_stride(arg249_1, (1920, ), (1, ))
    assert_size_stride(arg250_1, (1920, ), (1, ))
    assert_size_stride(arg251_1, (1920, ), (1, ))
    assert_size_stride(arg252_1, (1920, ), (1, ))
    assert_size_stride(arg253_1, (640, ), (1, ))
    assert_size_stride(arg254_1, (640, ), (1, ))
    assert_size_stride(arg255_1, (1920, ), (1, ))
    assert_size_stride(arg256_1, (1920, ), (1, ))
    assert_size_stride(arg257_1, (1920, ), (1, ))
    assert_size_stride(arg258_1, (1920, ), (1, ))
    assert_size_stride(arg259_1, (640, ), (1, ))
    assert_size_stride(arg260_1, (640, ), (1, ))
    assert_size_stride(arg261_1, (1920, ), (1, ))
    assert_size_stride(arg262_1, (1920, ), (1, ))
    assert_size_stride(arg263_1, (1920, ), (1, ))
    assert_size_stride(arg264_1, (1920, ), (1, ))
    assert_size_stride(arg265_1, (640, ), (1, ))
    assert_size_stride(arg266_1, (640, ), (1, ))
    assert_size_stride(arg267_1, (1920, ), (1, ))
    assert_size_stride(arg268_1, (1920, ), (1, ))
    assert_size_stride(arg269_1, (1920, ), (1, ))
    assert_size_stride(arg270_1, (1920, ), (1, ))
    assert_size_stride(arg271_1, (640, ), (1, ))
    assert_size_stride(arg272_1, (640, ), (1, ))
    assert_size_stride(arg273_1, (1920, ), (1, ))
    assert_size_stride(arg274_1, (1920, ), (1, ))
    assert_size_stride(arg275_1, (1920, ), (1, ))
    assert_size_stride(arg276_1, (1920, ), (1, ))
    assert_size_stride(arg277_1, (640, ), (1, ))
    assert_size_stride(arg278_1, (640, ), (1, ))
    assert_size_stride(arg279_1, (1920, ), (1, ))
    assert_size_stride(arg280_1, (1920, ), (1, ))
    assert_size_stride(arg281_1, (1920, ), (1, ))
    assert_size_stride(arg282_1, (1920, ), (1, ))
    assert_size_stride(arg283_1, (640, ), (1, ))
    assert_size_stride(arg284_1, (640, ), (1, ))
    assert_size_stride(arg285_1, (2560, ), (1, ))
    assert_size_stride(arg286_1, (2560, ), (1, ))
    assert_size_stride(arg287_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    buf0 = empty_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg287_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg114_1
    del arg287_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 128, 128), (524288, 1, 4096, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    del arg115_1
    del arg173_1
    del arg174_1
    del arg1_1
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf5 = extern_kernels.convolution(buf3, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf5, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del buf4
    buf6 = buf5; del buf5  # reuse
    buf7 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf7.data_ptr()))
    del arg116_1
    del arg175_1
    del arg176_1
    del arg2_1
    del arg3_1
    # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del buf6
    del buf7
    # Source Nodes: [x_20], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf3, arg117_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 128, 64, 64), (524288, 1, 8192, 128))
    del arg117_1
    del buf3
    buf10 = buf8; del buf8  # reuse
    buf11 = buf10; del buf10  # reuse
    buf12 = empty_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_3(c_void_p(buf11.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg118_1
    del arg177_1
    del arg178_1
    del arg179_1
    del arg180_1
    del arg4_1
    del arg5_1
    del arg6_1
    del arg7_1
    del buf9
    # Source Nodes: [shortcut_1, x_26], Original ATen: [aten.convolution, aten.relu]
    buf13 = extern_kernels.convolution(buf11, buf12, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf13, (8, 192, 32, 32), (196608, 1, 6144, 192))
    del buf12
    buf14 = buf13; del buf13  # reuse
    buf15 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_4(c_void_p(buf14.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg119_1
    del arg181_1
    del arg182_1
    del arg8_1
    del arg9_1
    # Source Nodes: [x_27, x_31, x_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (8, 192, 32, 32), (196608, 1, 6144, 192))
    del buf14
    # Source Nodes: [x_40], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(buf11, arg120_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf17, (8, 192, 32, 32), (196608, 1, 6144, 192))
    del arg120_1
    del buf11
    buf18 = buf16; del buf16  # reuse
    buf19 = buf18; del buf18  # reuse
    buf20 = buf15; del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_relu_5(c_void_p(buf19.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(buf20.data_ptr()))
    del arg10_1
    del arg11_1
    del arg121_1
    del arg12_1
    del arg13_1
    del arg183_1
    del arg184_1
    del arg185_1
    del arg186_1
    del buf17
    # Source Nodes: [shortcut_2, x_46], Original ATen: [aten.convolution, aten.relu]
    buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 192, 32, 32), (196608, 1, 6144, 192))
    buf22 = buf21; del buf21  # reuse
    buf23 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_6(c_void_p(buf22.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg122_1
    del arg14_1
    del arg15_1
    del arg187_1
    del arg188_1
    # Source Nodes: [x_47, x_51, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf24 = extern_kernels.convolution(buf22, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 192, 32, 32), (196608, 1, 6144, 192))
    del buf22
    del buf23
    buf25 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_7(c_void_p(buf25.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg189_1
    del arg190_1
    del buf24
    # Source Nodes: [x_61], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(buf25, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 160, 32, 32), (163840, 1, 5120, 160))
    del arg123_1
    buf27 = buf26; del buf26  # reuse
    buf28 = empty_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_8(c_void_p(buf27.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg124_1
    del arg18_1
    del arg191_1
    del arg192_1
    del arg19_1
    # Source Nodes: [x_62, x_66, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf29 = extern_kernels.convolution(buf27, buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf27
    buf30 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf30.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg193_1
    del arg194_1
    del arg20_1
    del arg21_1
    # Source Nodes: [x_68, x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf31 = extern_kernels.convolution(buf30, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg125_1
    del buf30
    # Source Nodes: [x_83], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf25, arg126_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg126_1
    del buf25
    buf33 = buf31; del buf31  # reuse
    buf34 = buf33; del buf33  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_10(c_void_p(buf34.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg195_1
    del arg196_1
    del arg197_1
    del arg198_1
    del arg22_1
    del arg23_1
    del arg24_1
    del arg25_1
    del buf32
    # Source Nodes: [shortcut_4, x_89], Original ATen: [aten.convolution, aten.relu]
    buf35 = extern_kernels.convolution(buf34, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf35, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del arg127_1
    buf36 = buf35; del buf35  # reuse
    buf37 = buf28; del buf28  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11(c_void_p(buf36.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg128_1.data_ptr()), c_void_p(buf37.data_ptr()))
    del arg128_1
    del arg199_1
    del arg200_1
    del arg26_1
    del arg27_1
    # Source Nodes: [x_90, x_94, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf38 = extern_kernels.convolution(buf36, buf37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf38, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf36
    buf39 = buf38; del buf38  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_12(c_void_p(buf39.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg201_1
    del arg202_1
    del arg28_1
    del arg29_1
    # Source Nodes: [x_100, x_103, x_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf40 = extern_kernels.convolution(buf39, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf40, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg129_1
    del buf39
    buf41 = buf34; del buf34  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_13(c_void_p(buf41.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg203_1
    del arg204_1
    del arg30_1
    del arg31_1
    del buf40
    # Source Nodes: [x_112], Original ATen: [aten.convolution]
    buf42 = extern_kernels.convolution(buf41, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del arg130_1
    buf43 = buf42; del buf42  # reuse
    buf44 = buf37; del buf37  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_14(c_void_p(buf43.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(arg131_1.data_ptr()), c_void_p(buf44.data_ptr()))
    del arg131_1
    del arg205_1
    del arg206_1
    del arg32_1
    del arg33_1
    # Source Nodes: [x_113, x_117, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf45 = extern_kernels.convolution(buf43, buf44, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf43
    buf46 = buf45; del buf45  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf46.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg207_1
    del arg208_1
    del arg34_1
    del arg35_1
    # Source Nodes: [x_119, x_123, x_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf46, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg132_1
    del buf46
    buf48 = buf41; del buf41  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf48.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg209_1
    del arg210_1
    del arg36_1
    del arg37_1
    del buf47
    # Source Nodes: [x_135], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf48, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del arg133_1
    buf50 = buf49; del buf49  # reuse
    buf51 = buf44; del buf44  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf50.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(arg134_1.data_ptr()), c_void_p(buf51.data_ptr()))
    del arg134_1
    del arg211_1
    del arg212_1
    del arg38_1
    del arg39_1
    # Source Nodes: [x_136, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf52 = extern_kernels.convolution(buf50, buf51, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf50
    buf53 = buf52; del buf52  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf53.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg213_1
    del arg214_1
    del arg40_1
    del arg41_1
    # Source Nodes: [x_142, x_146, x_149], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf54 = extern_kernels.convolution(buf53, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg135_1
    del buf53
    buf55 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_19(c_void_p(buf55.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg215_1
    del arg216_1
    del arg42_1
    del arg43_1
    del buf54
    # Source Nodes: [x_158], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf56, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del arg136_1
    buf57 = buf56; del buf56  # reuse
    buf58 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_20(c_void_p(buf57.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg137_1.data_ptr()), c_void_p(buf58.data_ptr()))
    del arg137_1
    del arg217_1
    del arg218_1
    del arg44_1
    del arg45_1
    # Source Nodes: [x_159, x_163, x_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf59 = extern_kernels.convolution(buf57, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf59, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf57
    buf60 = buf59; del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf60.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg219_1
    del arg220_1
    del arg46_1
    del arg47_1
    # Source Nodes: [x_165, x_169, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf61 = extern_kernels.convolution(buf60, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf61, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg138_1
    del buf60
    buf62 = buf55; del buf55  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_22(c_void_p(buf62.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg221_1
    del arg222_1
    del arg48_1
    del arg49_1
    del buf61
    # Source Nodes: [x_181], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del arg139_1
    buf64 = buf63; del buf63  # reuse
    buf65 = buf58; del buf58  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_23(c_void_p(buf64.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg140_1.data_ptr()), c_void_p(buf65.data_ptr()))
    del arg140_1
    del arg223_1
    del arg224_1
    del arg50_1
    del arg51_1
    # Source Nodes: [x_182, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf66 = extern_kernels.convolution(buf64, buf65, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (8, 160, 16, 16), (40960, 1, 2560, 160))
    del buf64
    del buf65
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_24(c_void_p(buf67.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg225_1
    del arg226_1
    del arg52_1
    del arg53_1
    # Source Nodes: [x_188, x_192, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf68 = extern_kernels.convolution(buf67, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 640, 16, 16), (163840, 1, 10240, 640))
    del arg141_1
    del buf67
    buf69 = buf62; del buf62  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_25(c_void_p(buf69.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()))
    del arg227_1
    del arg228_1
    del arg54_1
    del arg55_1
    del buf68
    # Source Nodes: [x_204], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    del arg142_1
    buf71 = buf70; del buf70  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_26(c_void_p(buf71.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()))
    del arg229_1
    del arg230_1
    del arg56_1
    del arg57_1
    # Source Nodes: [x_205, x_209, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf72 = extern_kernels.convolution(buf71, arg143_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf72, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg143_1
    del buf71
    buf73 = buf72; del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_27(c_void_p(buf73.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg231_1
    del arg232_1
    del arg58_1
    del arg59_1
    # Source Nodes: [x_211, x_215, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf74 = extern_kernels.convolution(buf73, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg144_1
    del buf73
    # Source Nodes: [x_226], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf69, arg145_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf75, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg145_1
    del buf69
    buf76 = buf74; del buf74  # reuse
    buf77 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_28(c_void_p(buf77.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()))
    del arg233_1
    del arg234_1
    del arg235_1
    del arg236_1
    del arg60_1
    del arg61_1
    del arg62_1
    del arg63_1
    del buf75
    # Source Nodes: [shortcut_10, x_232], Original ATen: [aten.convolution, aten.relu]
    buf78 = extern_kernels.convolution(buf77, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf78, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg146_1
    buf79 = buf78; del buf78  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_29(c_void_p(buf79.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg237_1
    del arg238_1
    del arg64_1
    del arg65_1
    # Source Nodes: [x_233, x_237, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf80 = extern_kernels.convolution(buf79, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf80, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg147_1
    del buf79
    buf81 = buf80; del buf80  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_30(c_void_p(buf81.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg239_1
    del arg240_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_239, x_243, x_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf82 = extern_kernels.convolution(buf81, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg148_1
    del buf81
    buf83 = buf77; del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_31(c_void_p(buf83.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg241_1
    del arg242_1
    del arg68_1
    del arg69_1
    del buf82
    # Source Nodes: [x_255], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg149_1
    buf85 = buf84; del buf84  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_32(c_void_p(buf85.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg243_1
    del arg244_1
    del arg70_1
    del arg71_1
    # Source Nodes: [x_256, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf86 = extern_kernels.convolution(buf85, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf86, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg150_1
    del buf85
    buf87 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_33(c_void_p(buf87.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()))
    del arg245_1
    del arg246_1
    del arg72_1
    del arg73_1
    # Source Nodes: [x_262, x_266, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf88 = extern_kernels.convolution(buf87, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg151_1
    del buf87
    buf89 = buf83; del buf83  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_34(c_void_p(buf89.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()))
    del arg247_1
    del arg248_1
    del arg74_1
    del arg75_1
    del buf88
    # Source Nodes: [x_278], Original ATen: [aten.convolution]
    buf90 = extern_kernels.convolution(buf89, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg152_1
    buf91 = buf90; del buf90  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_35(c_void_p(buf91.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg249_1
    del arg250_1
    del arg76_1
    del arg77_1
    # Source Nodes: [x_279, x_283, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf92 = extern_kernels.convolution(buf91, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf92, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg153_1
    del buf91
    buf93 = buf92; del buf92  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_36(c_void_p(buf93.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()))
    del arg251_1
    del arg252_1
    del arg78_1
    del arg79_1
    # Source Nodes: [x_285, x_289, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf94 = extern_kernels.convolution(buf93, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf94, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg154_1
    del buf93
    buf95 = buf89; del buf89  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_37(c_void_p(buf95.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()))
    del arg253_1
    del arg254_1
    del arg80_1
    del arg81_1
    del buf94
    # Source Nodes: [x_301], Original ATen: [aten.convolution]
    buf96 = extern_kernels.convolution(buf95, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf96, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg155_1
    buf97 = buf96; del buf96  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_38(c_void_p(buf97.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg255_1
    del arg256_1
    del arg82_1
    del arg83_1
    # Source Nodes: [x_302, x_306, x_307], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf98 = extern_kernels.convolution(buf97, arg156_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf98, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg156_1
    del buf97
    buf99 = buf98; del buf98  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_39(c_void_p(buf99.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()))
    del arg257_1
    del arg258_1
    del arg84_1
    del arg85_1
    # Source Nodes: [x_308, x_312, x_315], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf100 = extern_kernels.convolution(buf99, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf100, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg157_1
    del buf99
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_40(c_void_p(buf101.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf95.data_ptr()))
    del arg259_1
    del arg260_1
    del arg86_1
    del arg87_1
    del buf95
    # Source Nodes: [x_324], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg158_1
    buf103 = buf102; del buf102  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_41(c_void_p(buf103.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg261_1
    del arg262_1
    del arg88_1
    del arg89_1
    # Source Nodes: [x_325, x_329, x_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf104 = extern_kernels.convolution(buf103, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf104, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg159_1
    del buf103
    buf105 = buf104; del buf104  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_42(c_void_p(buf105.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()))
    del arg263_1
    del arg264_1
    del arg90_1
    del arg91_1
    # Source Nodes: [x_331, x_335, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf106 = extern_kernels.convolution(buf105, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg160_1
    del buf105
    buf107 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_43(c_void_p(buf107.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()))
    del arg265_1
    del arg266_1
    del arg92_1
    del arg93_1
    del buf106
    # Source Nodes: [x_347], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf107, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf108, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg161_1
    buf109 = buf108; del buf108  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_44(c_void_p(buf109.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg267_1
    del arg268_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_348, x_352, x_353], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf110 = extern_kernels.convolution(buf109, arg162_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf110, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg162_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_45(c_void_p(buf111.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()))
    del arg269_1
    del arg270_1
    del arg96_1
    del arg97_1
    # Source Nodes: [x_354, x_358, x_361], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf112 = extern_kernels.convolution(buf111, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf112, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg163_1
    del buf111
    buf113 = buf107; del buf107  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_46(c_void_p(buf113.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()))
    del arg271_1
    del arg272_1
    del arg98_1
    del arg99_1
    del buf112
    # Source Nodes: [x_370], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf113, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf114, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg164_1
    buf115 = buf114; del buf114  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_47(c_void_p(buf115.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg273_1
    del arg274_1
    # Source Nodes: [x_371, x_375, x_376], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf116 = extern_kernels.convolution(buf115, arg165_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf116, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg165_1
    del buf115
    buf117 = buf116; del buf116  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_48(c_void_p(buf117.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()))
    del arg102_1
    del arg103_1
    del arg275_1
    del arg276_1
    # Source Nodes: [x_377, x_381, x_384], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf118 = extern_kernels.convolution(buf117, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf118, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg166_1
    del buf117
    buf119 = buf113; del buf113  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_49(c_void_p(buf119.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()))
    del arg104_1
    del arg105_1
    del arg277_1
    del arg278_1
    del buf118
    # Source Nodes: [x_393], Original ATen: [aten.convolution]
    buf120 = extern_kernels.convolution(buf119, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf120, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg167_1
    buf121 = buf120; del buf120  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_50(c_void_p(buf121.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg279_1
    del arg280_1
    # Source Nodes: [x_394, x_398, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf122 = extern_kernels.convolution(buf121, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1920, bias=None)
    assert_size_stride(buf122, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    del arg168_1
    del buf121
    buf123 = buf122; del buf122  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_51(c_void_p(buf123.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()))
    del arg108_1
    del arg109_1
    del arg281_1
    del arg282_1
    # Source Nodes: [x_400, x_404, x_407], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf124 = extern_kernels.convolution(buf123, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 640, 8, 8), (40960, 1, 5120, 640))
    del arg169_1
    del buf123
    buf125 = buf119; del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_52(c_void_p(buf125.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()))
    del arg110_1
    del arg111_1
    del arg283_1
    del arg284_1
    del buf124
    # Source Nodes: [x_408, x_415, x_416, x_417], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution, aten.relu]
    buf126 = extern_kernels.convolution(buf125, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    del arg170_1
    del buf125
    buf127 = empty_strided((8, 2560, 1, 1), (2560, 1, 20480, 20480), device='cpu', dtype=torch.float32)
    buf128 = reinterpret_tensor(buf127, (8, 2560, 1, 1), (2560, 1, 1, 1), 0); del buf127  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_53(c_void_p(buf128.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg285_1
    del arg286_1
    del buf126
    buf129 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_428], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg172_1, reinterpret_tensor(buf128, (8, 2560), (2560, 1), 0), reinterpret_tensor(arg171_1, (2560, 1000), (1, 2560), 0), alpha=1, beta=1, out=buf129)
    del arg171_1
    del arg172_1
    return (buf129, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((1000, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
