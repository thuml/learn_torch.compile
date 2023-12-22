
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x2 + (613760L*x1) + (1841280L*x0))];
                        out_ptr0[static_cast<long>(x1 + (3L*x2) + (1841280L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (64L*x2) + (39280640L*x0)), static_cast<long>(64L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(479L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((2L*x3) + (1918L*x2) + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(1L + (2L*x3) + (1918L*x2) + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(959L + (2L*x3) + (1918L*x2) + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(960L + (2L*x3) + (1918L*x2) + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            tmp6.store(out_ptr1 + static_cast<long>(x1 + (64L*x3) + (30656L*x2) + (9809920L*x0)));
                        }
                    }
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr2 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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


cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(153280L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (128L*x2) + (19619840L*x0)), static_cast<long>(128L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(239L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((2L*x3) + (958L*x2) + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(1L + (2L*x3) + (958L*x2) + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(479L + (2L*x3) + (958L*x2) + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(480L + (2L*x3) + (958L*x2) + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            tmp6.store(out_ptr1 + static_cast<long>(x1 + (128L*x3) + (30592L*x2) + (4894720L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38240L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (256L*x2) + (9789440L*x0)), static_cast<long>(256L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(119L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((2L*x3) + (478L*x2) + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(1L + (2L*x3) + (478L*x2) + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(239L + (2L*x3) + (478L*x2) + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(240L + (2L*x3) + (478L*x2) + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            tmp6.store(out_ptr1 + static_cast<long>(x1 + (256L*x3) + (30464L*x2) + (2437120L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9520L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (512L*x2) + (4874240L*x0)), static_cast<long>(512L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = in_ptr1[static_cast<long>(x1 + x1_inner)];
                            auto tmp5 = in_ptr2[static_cast<long>(x1 + x1_inner)];
                            auto tmp14 = in_ptr3[static_cast<long>(x1 + x1_inner)];
                            auto tmp17 = in_ptr4[static_cast<long>(x1 + x1_inner)];
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
                            auto tmp20 = at::vec::clamp_min(tmp19, decltype(tmp19)(0));
                            tmp20.store(out_ptr0 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(40L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(59L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>((2L*x3) + (238L*x2) + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(1L + (2L*x3) + (238L*x2) + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp3 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(119L + (2L*x3) + (238L*x2) + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp5 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(120L + (2L*x3) + (238L*x2) + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            auto tmp2 = at::vec::maximum(tmp1, tmp0);
                            auto tmp4 = at::vec::maximum(tmp3, tmp2);
                            auto tmp6 = at::vec::maximum(tmp5, tmp4);
                            tmp6.store(out_ptr1 + static_cast<long>(x1 + (512L*x3) + (30208L*x2) + (1208320L*x0)));
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4720L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4720L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(118L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4936708860759494);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.49572649572649574);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = c10::convert<long>(tmp16);
                            auto tmp18 = in_out_ptr0[static_cast<long>(x1 + (512L*tmp17) + (30208L*tmp9) + (1208320L*x0))];
                            auto tmp19 = c10::convert<float>(tmp9);
                            auto tmp20 = decltype(tmp8)(tmp8 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp18)(tmp18 * tmp22);
                            out_ptr0[static_cast<long>(x3 + (118L*x2) + (9440L*x1) + (4833280L*x0))] = tmp23;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(118L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4936708860759494);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(39.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.49572649572649574);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (512L*tmp20) + (30208L*tmp12) + (1208320L*x0))];
                            auto tmp22 = c10::convert<long>(tmp8);
                            auto tmp23 = c10::convert<float>(tmp22);
                            auto tmp24 = decltype(tmp8)(tmp8 - tmp23);
                            auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                            out_ptr1[static_cast<long>(x3 + (118L*x2) + (9440L*x1) + (4833280L*x0))] = tmp25;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(118L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4936708860759494);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.49572649572649574);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = std::ceil(tmp16);
                            auto tmp18 = static_cast<float>(58.0);
                            auto tmp19 = min_propagate_nan(tmp17, tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (512L*tmp20) + (30208L*tmp9) + (1208320L*x0))];
                            auto tmp22 = c10::convert<float>(tmp9);
                            auto tmp23 = decltype(tmp8)(tmp8 - tmp22);
                            auto tmp24 = static_cast<float>(1.0);
                            auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                            auto tmp26 = decltype(tmp21)(tmp21 * tmp25);
                            out_ptr2[static_cast<long>(x3 + (118L*x2) + (9440L*x1) + (4833280L*x0))] = tmp26;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(80L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(118L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4936708860759494);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(39.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.49572649572649574);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::ceil(tmp19);
                            auto tmp21 = static_cast<float>(58.0);
                            auto tmp22 = min_propagate_nan(tmp20, tmp21);
                            auto tmp23 = c10::convert<long>(tmp22);
                            auto tmp24 = in_out_ptr0[static_cast<long>(x1 + (512L*tmp23) + (30208L*tmp12) + (1208320L*x0))];
                            auto tmp25 = c10::convert<long>(tmp8);
                            auto tmp26 = c10::convert<float>(tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (118L*x2) + (9440L*x1) + (4833280L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(118L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (118L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x1 + (118L*x0))];
                    auto tmp18 = out_ptr2[static_cast<long>(x1 + (118L*x0))];
                    auto tmp19 = out_ptr3[static_cast<long>(x1 + (118L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = c10::convert<long>(x1);
                    auto tmp4 = c10::convert<double>(tmp3);
                    auto tmp5 = static_cast<double>(1.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = static_cast<double>(0.0);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = c10::convert<float>(tmp8);
                    auto tmp10 = static_cast<float>(0.49572649572649574);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<float>(tmp12);
                    auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                    auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                    auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                    in_out_ptr1[static_cast<long>(x1 + (118L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40960L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(119L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(118);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr1[static_cast<long>(x2 + (118L*x1) + (4833280L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr4[static_cast<long>(x2 + (119L*x1) + (9748480L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(9520L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9520L*x1) + (9520L*x1_inner) + (9748480L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (1024L*x2) + (9748480L*x0)), static_cast<long>(1024L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)), static_cast<long>(1024L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (9216L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (1024L*x2) + (9216L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(19040L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(238L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4968553459119497);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4978902953586498);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = c10::convert<long>(tmp16);
                            auto tmp18 = in_out_ptr0[static_cast<long>(x1 + (256L*tmp17) + (30464L*tmp9) + (2437120L*x0))];
                            auto tmp19 = c10::convert<float>(tmp9);
                            auto tmp20 = decltype(tmp8)(tmp8 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp18)(tmp18 * tmp22);
                            out_ptr0[static_cast<long>(x3 + (238L*x2) + (38080L*x1) + (9748480L*x0))] = tmp23;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(238L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4968553459119497);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(79.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4978902953586498);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (256L*tmp20) + (30464L*tmp12) + (2437120L*x0))];
                            auto tmp22 = c10::convert<long>(tmp8);
                            auto tmp23 = c10::convert<float>(tmp22);
                            auto tmp24 = decltype(tmp8)(tmp8 - tmp23);
                            auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                            out_ptr1[static_cast<long>(x3 + (238L*x2) + (38080L*x1) + (9748480L*x0))] = tmp25;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(238L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4968553459119497);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4978902953586498);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = std::ceil(tmp16);
                            auto tmp18 = static_cast<float>(118.0);
                            auto tmp19 = min_propagate_nan(tmp17, tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (256L*tmp20) + (30464L*tmp9) + (2437120L*x0))];
                            auto tmp22 = c10::convert<float>(tmp9);
                            auto tmp23 = decltype(tmp8)(tmp8 - tmp22);
                            auto tmp24 = static_cast<float>(1.0);
                            auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                            auto tmp26 = decltype(tmp21)(tmp21 * tmp25);
                            out_ptr2[static_cast<long>(x3 + (238L*x2) + (38080L*x1) + (9748480L*x0))] = tmp26;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(160L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(238L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.4968553459119497);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(79.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4978902953586498);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::ceil(tmp19);
                            auto tmp21 = static_cast<float>(118.0);
                            auto tmp22 = min_propagate_nan(tmp20, tmp21);
                            auto tmp23 = c10::convert<long>(tmp22);
                            auto tmp24 = in_out_ptr0[static_cast<long>(x1 + (256L*tmp23) + (30464L*tmp12) + (2437120L*x0))];
                            auto tmp25 = c10::convert<long>(tmp8);
                            auto tmp26 = c10::convert<float>(tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (238L*x2) + (38080L*x1) + (9748480L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(238L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (238L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x1 + (238L*x0))];
                    auto tmp18 = out_ptr2[static_cast<long>(x1 + (238L*x0))];
                    auto tmp19 = out_ptr3[static_cast<long>(x1 + (238L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = c10::convert<long>(x1);
                    auto tmp4 = c10::convert<double>(tmp3);
                    auto tmp5 = static_cast<double>(1.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = static_cast<double>(0.0);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = c10::convert<float>(tmp8);
                    auto tmp10 = static_cast<float>(0.4978902953586498);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<float>(tmp12);
                    auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                    auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                    auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                    in_out_ptr1[static_cast<long>(x1 + (238L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40960L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(239L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(238);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr1[static_cast<long>(x2 + (238L*x1) + (9748480L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr4[static_cast<long>(x2 + (239L*x1) + (19578880L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(38240L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (38240L*x1) + (38240L*x1_inner) + (19578880L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (512L*x2) + (19578880L*x0)), static_cast<long>(512L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)), static_cast<long>(512L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (4608L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (512L*x2) + (4608L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
                    auto tmp17 = at::vec::clamp_min(tmp16, decltype(tmp16)(0));
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(76480L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(478L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49843260188087773);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4989517819706499);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = c10::convert<long>(tmp16);
                            auto tmp18 = in_out_ptr0[static_cast<long>(x1 + (128L*tmp17) + (30592L*tmp9) + (4894720L*x0))];
                            auto tmp19 = c10::convert<float>(tmp9);
                            auto tmp20 = decltype(tmp8)(tmp8 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp18)(tmp18 * tmp22);
                            out_ptr0[static_cast<long>(x3 + (478L*x2) + (152960L*x1) + (19578880L*x0))] = tmp23;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(478L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49843260188087773);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(159.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4989517819706499);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (128L*tmp20) + (30592L*tmp12) + (4894720L*x0))];
                            auto tmp22 = c10::convert<long>(tmp8);
                            auto tmp23 = c10::convert<float>(tmp22);
                            auto tmp24 = decltype(tmp8)(tmp8 - tmp23);
                            auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                            out_ptr1[static_cast<long>(x3 + (478L*x2) + (152960L*x1) + (19578880L*x0))] = tmp25;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(478L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49843260188087773);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4989517819706499);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = std::ceil(tmp16);
                            auto tmp18 = static_cast<float>(238.0);
                            auto tmp19 = min_propagate_nan(tmp17, tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (128L*tmp20) + (30592L*tmp9) + (4894720L*x0))];
                            auto tmp22 = c10::convert<float>(tmp9);
                            auto tmp23 = decltype(tmp8)(tmp8 - tmp22);
                            auto tmp24 = static_cast<float>(1.0);
                            auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                            auto tmp26 = decltype(tmp21)(tmp21 * tmp25);
                            out_ptr2[static_cast<long>(x3 + (478L*x2) + (152960L*x1) + (19578880L*x0))] = tmp26;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(320L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(478L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49843260188087773);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(159.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4989517819706499);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::ceil(tmp19);
                            auto tmp21 = static_cast<float>(238.0);
                            auto tmp22 = min_propagate_nan(tmp20, tmp21);
                            auto tmp23 = c10::convert<long>(tmp22);
                            auto tmp24 = in_out_ptr0[static_cast<long>(x1 + (128L*tmp23) + (30592L*tmp12) + (4894720L*x0))];
                            auto tmp25 = c10::convert<long>(tmp8);
                            auto tmp26 = c10::convert<float>(tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (478L*x2) + (152960L*x1) + (19578880L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(478L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (478L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x1 + (478L*x0))];
                    auto tmp18 = out_ptr2[static_cast<long>(x1 + (478L*x0))];
                    auto tmp19 = out_ptr3[static_cast<long>(x1 + (478L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = c10::convert<long>(x1);
                    auto tmp4 = c10::convert<double>(tmp3);
                    auto tmp5 = static_cast<double>(1.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = static_cast<double>(0.0);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = c10::convert<float>(tmp8);
                    auto tmp10 = static_cast<float>(0.4989517819706499);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<float>(tmp12);
                    auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                    auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                    auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                    in_out_ptr1[static_cast<long>(x1 + (478L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40960L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(479L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(478);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr1[static_cast<long>(x2 + (478L*x1) + (19578880L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr4[static_cast<long>(x2 + (479L*x1) + (39239680L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(153280L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (153280L*x1) + (153280L*x1_inner) + (39239680L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (256L*x2) + (39239680L*x0)), static_cast<long>(256L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(1L))
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
}
''')


cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(306560L); x0+=static_cast<long>(1L))
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(958L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49921752738654146);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4994775339602926);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = c10::convert<long>(tmp16);
                            auto tmp18 = in_out_ptr0[static_cast<long>(x1 + (64L*tmp17) + (30656L*tmp9) + (9809920L*x0))];
                            auto tmp19 = c10::convert<float>(tmp9);
                            auto tmp20 = decltype(tmp8)(tmp8 - tmp19);
                            auto tmp21 = static_cast<float>(1.0);
                            auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                            auto tmp23 = decltype(tmp18)(tmp18 * tmp22);
                            out_ptr0[static_cast<long>(x3 + (958L*x2) + (613120L*x1) + (39239680L*x0))] = tmp23;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(958L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49921752738654146);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(319.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4994775339602926);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (64L*tmp20) + (30656L*tmp12) + (9809920L*x0))];
                            auto tmp22 = c10::convert<long>(tmp8);
                            auto tmp23 = c10::convert<float>(tmp22);
                            auto tmp24 = decltype(tmp8)(tmp8 - tmp23);
                            auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                            out_ptr1[static_cast<long>(x3 + (958L*x2) + (613120L*x1) + (39239680L*x0))] = tmp25;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(958L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49921752738654146);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = c10::convert<long>(tmp8);
                            auto tmp10 = c10::convert<long>(x3);
                            auto tmp11 = c10::convert<double>(tmp10);
                            auto tmp12 = decltype(tmp11)(tmp11 * tmp2);
                            auto tmp13 = decltype(tmp12)(tmp12 + tmp4);
                            auto tmp14 = c10::convert<float>(tmp13);
                            auto tmp15 = static_cast<float>(0.4994775339602926);
                            auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                            auto tmp17 = std::ceil(tmp16);
                            auto tmp18 = static_cast<float>(478.0);
                            auto tmp19 = min_propagate_nan(tmp17, tmp18);
                            auto tmp20 = c10::convert<long>(tmp19);
                            auto tmp21 = in_out_ptr0[static_cast<long>(x1 + (64L*tmp20) + (30656L*tmp9) + (9809920L*x0))];
                            auto tmp22 = c10::convert<float>(tmp9);
                            auto tmp23 = decltype(tmp8)(tmp8 - tmp22);
                            auto tmp24 = static_cast<float>(1.0);
                            auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                            auto tmp26 = decltype(tmp21)(tmp21 * tmp25);
                            out_ptr2[static_cast<long>(x3 + (958L*x2) + (613120L*x1) + (39239680L*x0))] = tmp26;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(640L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(958L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = c10::convert<double>(tmp0);
                            auto tmp2 = static_cast<double>(1.0);
                            auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                            auto tmp4 = static_cast<double>(0.0);
                            auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                            auto tmp6 = c10::convert<float>(tmp5);
                            auto tmp7 = static_cast<float>(0.49921752738654146);
                            auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                            auto tmp9 = std::ceil(tmp8);
                            auto tmp10 = static_cast<float>(319.0);
                            auto tmp11 = min_propagate_nan(tmp9, tmp10);
                            auto tmp12 = c10::convert<long>(tmp11);
                            auto tmp13 = c10::convert<long>(x3);
                            auto tmp14 = c10::convert<double>(tmp13);
                            auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                            auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                            auto tmp17 = c10::convert<float>(tmp16);
                            auto tmp18 = static_cast<float>(0.4994775339602926);
                            auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                            auto tmp20 = std::ceil(tmp19);
                            auto tmp21 = static_cast<float>(478.0);
                            auto tmp22 = min_propagate_nan(tmp20, tmp21);
                            auto tmp23 = c10::convert<long>(tmp22);
                            auto tmp24 = in_out_ptr0[static_cast<long>(x1 + (64L*tmp23) + (30656L*tmp12) + (9809920L*x0))];
                            auto tmp25 = c10::convert<long>(tmp8);
                            auto tmp26 = c10::convert<float>(tmp25);
                            auto tmp27 = decltype(tmp8)(tmp8 - tmp26);
                            auto tmp28 = decltype(tmp24)(tmp24 * tmp27);
                            out_ptr3[static_cast<long>(x3 + (958L*x2) + (613120L*x1) + (39239680L*x0))] = tmp28;
                        }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(81920L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(958L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x1 + (958L*x0))];
                    auto tmp1 = out_ptr1[static_cast<long>(x1 + (958L*x0))];
                    auto tmp18 = out_ptr2[static_cast<long>(x1 + (958L*x0))];
                    auto tmp19 = out_ptr3[static_cast<long>(x1 + (958L*x0))];
                    auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                    auto tmp3 = c10::convert<long>(x1);
                    auto tmp4 = c10::convert<double>(tmp3);
                    auto tmp5 = static_cast<double>(1.0);
                    auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                    auto tmp7 = static_cast<double>(0.0);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = c10::convert<float>(tmp8);
                    auto tmp10 = static_cast<float>(0.4994775339602926);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<float>(tmp12);
                    auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                    auto tmp15 = static_cast<float>(1.0);
                    auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                    auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                    auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                    auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                    in_out_ptr1[static_cast<long>(x1 + (958L*x0))] = tmp22;
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40960L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(959L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = static_cast<long>(958);
                        auto tmp2 = tmp0 < tmp1;
                        auto tmp3 = [&]
                        {
                            auto tmp4 = in_out_ptr1[static_cast<long>(x2 + (958L*x1) + (39239680L*x0))];
                            return tmp4;
                        }
                        ;
                        auto tmp5 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                        out_ptr4[static_cast<long>(x2 + (959L*x1) + (78561280L*x0))] = tmp5;
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2 + (613760L*x1) + (613760L*x1_inner) + (78561280L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (128L*x2) + (78561280L*x0)), static_cast<long>(128L));
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                        {
                            float tmp1[8*8] __attribute__ ((aligned (8)));
                            for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                            {
                                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0)));
                                tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                            }
                            at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                        }
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                            tmp0.store(out_ptr6 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1227520L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(613760L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x1 + (2L*x2) + (1227520L*x0))];
                        out_ptr0[static_cast<long>(x2 + (613760L*x1) + (1227520L*x0))] = tmp0;
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg69_1, (64, ), (1, ))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (64, ), (1, ))
    assert_size_stride(arg72_1, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg73_1, (2, ), (1, ))
    assert_size_stride(arg74_1, (64, ), (1, ))
    assert_size_stride(arg75_1, (64, ), (1, ))
    assert_size_stride(arg76_1, (), ())
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (64, ), (1, ))
    assert_size_stride(arg79_1, (), ())
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (), ())
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (), ())
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (), ())
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (), ())
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (), ())
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (), ())
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (), ())
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (), ())
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (), ())
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (), ())
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (), ())
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (), ())
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (), ())
    assert_size_stride(arg119_1, (64, ), (1, ))
    assert_size_stride(arg120_1, (64, ), (1, ))
    assert_size_stride(arg121_1, (), ())
    assert_size_stride(arg122_1, (64, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (), ())
    assert_size_stride(arg125_1, (64, ), (1, ))
    assert_size_stride(arg126_1, (64, ), (1, ))
    assert_size_stride(arg127_1, (), ())
    assert_size_stride(arg128_1, (2, 3, 640, 959), (1841280, 613760, 959, 1))
    buf0 = empty_strided((2, 3, 640, 959), (1841280, 1, 2877, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg128_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg128_1
    # Source Nodes: [l__mod___inc_double_conv_0], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, arg1_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf2, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del arg1_1
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg2_1
    del arg3_1
    del arg4_1
    del arg74_1
    del arg75_1
    # Source Nodes: [l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf5 = extern_kernels.convolution(buf3, buf4, arg5_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf5, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del arg5_1
    del buf3
    buf83 = empty((2, 128, 640, 959), device='cpu', dtype=torch.float32)
    buf6 = reinterpret_tensor(buf83, (2, 64, 640, 959), (78561280, 613760, 959, 1), 0)  # alias
    buf7 = empty_strided((2, 64, 320, 479), (9809920, 1, 30656, 64), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg6_1
    del arg77_1
    del arg78_1
    del arg7_1
    del arg8_1
    del buf5
    # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, l__mod___down1_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
    buf9 = extern_kernels.convolution(buf7, buf8, arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf9, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del arg9_1
    del buf7
    buf10 = buf9; del buf9  # reuse
    buf11 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_3(c_void_p(buf10.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(buf11.data_ptr()))
    del arg10_1
    del arg11_1
    del arg12_1
    del arg80_1
    del arg81_1
    # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf12 = extern_kernels.convolution(buf10, buf11, arg13_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf12, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del arg13_1
    del buf10
    del buf11
    buf69 = empty((2, 256, 320, 479), device='cpu', dtype=torch.float32)
    buf13 = reinterpret_tensor(buf69, (2, 128, 320, 479), (39239680, 153280, 479, 1), 0)  # alias
    buf14 = empty_strided((2, 128, 160, 239), (4894720, 1, 30592, 128), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4(c_void_p(buf12.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del arg14_1
    del arg15_1
    del arg16_1
    del arg83_1
    del arg84_1
    del buf12
    # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, l__mod___down2_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
    buf16 = extern_kernels.convolution(buf14, buf15, arg17_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf16, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del arg17_1
    del buf14
    buf17 = buf16; del buf16  # reuse
    buf18 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_5(c_void_p(buf17.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg18_1
    del arg19_1
    del arg20_1
    del arg86_1
    del arg87_1
    # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf19 = extern_kernels.convolution(buf17, buf18, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf19, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del arg21_1
    del buf17
    del buf18
    buf55 = empty((2, 512, 160, 239), device='cpu', dtype=torch.float32)
    buf20 = reinterpret_tensor(buf55, (2, 256, 160, 239), (19578880, 38240, 239, 1), 0)  # alias
    buf21 = empty_strided((2, 256, 80, 119), (2437120, 1, 30464, 256), device='cpu', dtype=torch.float32)
    buf22 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6(c_void_p(buf19.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg22_1
    del arg23_1
    del arg24_1
    del arg89_1
    del arg90_1
    del buf19
    # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, l__mod___down3_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
    buf23 = extern_kernels.convolution(buf21, buf22, arg25_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf23, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del arg25_1
    del buf21
    buf24 = buf23; del buf23  # reuse
    buf25 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_7(c_void_p(buf24.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg26_1
    del arg27_1
    del arg28_1
    del arg92_1
    del arg93_1
    # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf24, buf25, arg29_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf26, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del arg29_1
    del buf24
    buf41 = empty((2, 1024, 80, 119), device='cpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf41, (2, 512, 80, 119), (9748480, 9520, 119, 1), 0)  # alias
    buf28 = empty_strided((2, 512, 40, 59), (1208320, 1, 30208, 512), device='cpu', dtype=torch.float32)
    buf29 = buf25; del buf25  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_8(c_void_p(buf26.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()))
    del arg30_1
    del arg31_1
    del arg32_1
    del arg95_1
    del arg96_1
    del buf26
    # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, l__mod___down4_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
    buf30 = extern_kernels.convolution(buf28, buf29, arg33_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf30, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    del arg33_1
    del buf28
    buf31 = buf30; del buf30  # reuse
    buf32 = buf29; del buf29  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_9(c_void_p(buf31.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(buf32.data_ptr()))
    del arg34_1
    del arg35_1
    del arg36_1
    del arg98_1
    del arg99_1
    # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_1, getattr_l__mod___down4_maxpool_conv___1___double_conv_2, getattr_l__mod___down4_maxpool_conv___1___double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf33 = extern_kernels.convolution(buf31, buf32, arg37_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf33, (2, 512, 40, 59), (1208320, 1, 30208, 512))
    del arg37_1
    del buf31
    del buf32
    buf34 = buf33; del buf33  # reuse
    buf35 = empty((2, 512, 80, 118), device='cpu', dtype=torch.float32)
    buf36 = empty((2, 512, 80, 118), device='cpu', dtype=torch.float32)
    buf37 = empty((2, 512, 80, 118), device='cpu', dtype=torch.float32)
    buf38 = empty((2, 512, 80, 118), device='cpu', dtype=torch.float32)
    buf39 = buf35; del buf35  # reuse
    buf40 = reinterpret_tensor(buf41, (2, 512, 80, 119), (9748480, 9520, 119, 1), 4874240)  # alias
    buf42 = empty_strided((2, 1024, 80, 119), (9748480, 1, 121856, 1024), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((512, 1024, 3, 3), (9216, 1, 3072, 1024), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_10(c_void_p(buf34.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg101_1
    del arg102_1
    del arg38_1
    del arg39_1
    del arg40_1
    del buf27
    del buf34
    del buf36
    del buf37
    del buf38
    del buf39
    del buf40
    # Source Nodes: [l__mod___up1_conv_double_conv_0], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf42, buf43, arg41_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (2, 512, 80, 119), (4874240, 1, 60928, 512))
    del arg41_1
    del buf43
    buf45 = buf44; del buf44  # reuse
    buf46 = reinterpret_tensor(buf22, (256, 512, 3, 3), (4608, 1, 1536, 512), 0); del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_11(c_void_p(buf45.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(buf46.data_ptr()))
    del arg104_1
    del arg105_1
    del arg42_1
    del arg43_1
    del arg44_1
    # Source Nodes: [l__mod___up1_conv_double_conv_1, l__mod___up1_conv_double_conv_2, l__mod___up1_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf47 = extern_kernels.convolution(buf45, buf46, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf47, (2, 256, 80, 119), (2437120, 1, 30464, 256))
    del arg45_1
    del buf45
    buf48 = buf47; del buf47  # reuse
    buf49 = reinterpret_tensor(buf42, (2, 256, 160, 238), (9748480, 38080, 238, 1), 0); del buf42  # reuse
    buf50 = reinterpret_tensor(buf41, (2, 256, 160, 238), (9748480, 38080, 238, 1), 0); del buf41  # reuse
    buf51 = empty((2, 256, 160, 238), device='cpu', dtype=torch.float32)
    buf52 = empty((2, 256, 160, 238), device='cpu', dtype=torch.float32)
    buf53 = buf49; del buf49  # reuse
    buf54 = reinterpret_tensor(buf55, (2, 256, 160, 239), (19578880, 38240, 239, 1), 9789440)  # alias
    buf56 = empty_strided((2, 512, 160, 239), (19578880, 1, 122368, 512), device='cpu', dtype=torch.float32)
    buf57 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_12(c_void_p(buf48.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del arg107_1
    del arg108_1
    del arg46_1
    del arg47_1
    del arg48_1
    del buf20
    del buf48
    del buf50
    del buf51
    del buf52
    del buf53
    del buf54
    # Source Nodes: [l__mod___up2_conv_double_conv_0], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf56, buf57, arg49_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (2, 256, 160, 239), (9789440, 1, 61184, 256))
    del arg49_1
    del buf57
    buf59 = buf58; del buf58  # reuse
    buf60 = reinterpret_tensor(buf15, (128, 256, 3, 3), (2304, 1, 768, 256), 0); del buf15  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_13(c_void_p(buf59.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(buf60.data_ptr()))
    del arg110_1
    del arg111_1
    del arg50_1
    del arg51_1
    del arg52_1
    # Source Nodes: [l__mod___up2_conv_double_conv_1, l__mod___up2_conv_double_conv_2, l__mod___up2_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf61 = extern_kernels.convolution(buf59, buf60, arg53_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf61, (2, 128, 160, 239), (4894720, 1, 30592, 128))
    del arg53_1
    del buf59
    buf62 = buf61; del buf61  # reuse
    buf63 = reinterpret_tensor(buf56, (2, 128, 320, 478), (19578880, 152960, 478, 1), 0); del buf56  # reuse
    buf64 = reinterpret_tensor(buf55, (2, 128, 320, 478), (19578880, 152960, 478, 1), 0); del buf55  # reuse
    buf65 = empty((2, 128, 320, 478), device='cpu', dtype=torch.float32)
    buf66 = empty((2, 128, 320, 478), device='cpu', dtype=torch.float32)
    buf67 = buf63; del buf63  # reuse
    buf68 = reinterpret_tensor(buf69, (2, 128, 320, 479), (39239680, 153280, 479, 1), 19619840)  # alias
    buf70 = empty_strided((2, 256, 320, 479), (39239680, 1, 122624, 256), device='cpu', dtype=torch.float32)
    buf71 = buf60; del buf60  # reuse
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_14(c_void_p(buf62.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()))
    del arg113_1
    del arg114_1
    del arg54_1
    del arg55_1
    del arg56_1
    del buf13
    del buf62
    del buf64
    del buf65
    del buf66
    del buf67
    del buf68
    # Source Nodes: [l__mod___up3_conv_double_conv_0], Original ATen: [aten.convolution]
    buf72 = extern_kernels.convolution(buf70, buf71, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf72, (2, 128, 320, 479), (19619840, 1, 61312, 128))
    del arg57_1
    del buf71
    buf73 = buf72; del buf72  # reuse
    buf74 = reinterpret_tensor(buf8, (64, 128, 3, 3), (1152, 1, 384, 128), 0); del buf8  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_15(c_void_p(buf73.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg116_1
    del arg117_1
    del arg58_1
    del arg59_1
    del arg60_1
    # Source Nodes: [l__mod___up3_conv_double_conv_1, l__mod___up3_conv_double_conv_2, l__mod___up3_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf75 = extern_kernels.convolution(buf73, buf74, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf75, (2, 64, 320, 479), (9809920, 1, 30656, 64))
    del arg61_1
    del buf73
    buf76 = buf75; del buf75  # reuse
    buf77 = reinterpret_tensor(buf70, (2, 64, 640, 958), (39239680, 613120, 958, 1), 0); del buf70  # reuse
    buf78 = reinterpret_tensor(buf69, (2, 64, 640, 958), (39239680, 613120, 958, 1), 0); del buf69  # reuse
    buf79 = empty((2, 64, 640, 958), device='cpu', dtype=torch.float32)
    buf80 = empty((2, 64, 640, 958), device='cpu', dtype=torch.float32)
    buf81 = buf77; del buf77  # reuse
    buf82 = reinterpret_tensor(buf83, (2, 64, 640, 959), (78561280, 613760, 959, 1), 39280640)  # alias
    buf84 = empty_strided((2, 128, 640, 959), (78561280, 1, 122752, 128), device='cpu', dtype=torch.float32)
    buf85 = buf74; del buf74  # reuse
    cpp_fused__native_batch_norm_legit_no_training__to_copy__unsafe_index_add_arange_constant_pad_nd_convolution_mul_relu_rsub_sub_16(c_void_p(buf76.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del arg119_1
    del arg120_1
    del arg62_1
    del arg63_1
    del arg64_1
    del buf6
    del buf76
    del buf78
    del buf79
    del buf80
    del buf81
    del buf82
    del buf83
    # Source Nodes: [l__mod___up4_conv_double_conv_0], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf84, buf85, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf86, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del arg65_1
    del buf84
    del buf85
    buf87 = buf86; del buf86  # reuse
    buf88 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_17(c_void_p(buf87.data_ptr()), c_void_p(arg122_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg122_1
    del arg123_1
    del arg66_1
    del arg67_1
    del arg68_1
    # Source Nodes: [l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf89 = extern_kernels.convolution(buf87, buf88, arg69_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf89, (2, 64, 640, 959), (39280640, 1, 61376, 64))
    del arg69_1
    del buf87
    del buf88
    buf90 = buf89; del buf89  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_18(c_void_p(buf90.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg125_1
    del arg126_1
    del arg70_1
    del arg71_1
    # Source Nodes: [l__mod___up4_conv_double_conv_4, logits, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf91 = extern_kernels.convolution(buf90, arg72_1, arg73_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf91, (2, 2, 640, 959), (1227520, 1, 1918, 2))
    del arg72_1
    del arg73_1
    del buf90
    buf92 = empty((2, 2, 640, 959), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_relu_19(c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    return (buf92, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((2, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg77_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg80_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg83_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg86_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg89_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg92_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg95_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg98_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg101_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg104_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg107_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg110_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg113_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg116_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg119_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg122_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg125_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cpu', dtype=torch.int64)
    arg128_1 = rand_strided((2, 3, 640, 959), (1841280, 613760, 959, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pytorch_unet', benchmark_compiled_module)
