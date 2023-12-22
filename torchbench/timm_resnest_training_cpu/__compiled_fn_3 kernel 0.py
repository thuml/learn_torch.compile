
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
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr1[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr1 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                {
                    float tmp1[8*8] __attribute__ ((aligned (8)));
                    for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr3 + static_cast<long>(x1 + (32L*x2) + (288L*x0)), static_cast<long>(32L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr3[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (288L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr3 + static_cast<long>(x1 + (32L*x2) + (288L*x0)));
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
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
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr4 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr4[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr4 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
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
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr5 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)), static_cast<long>(128L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr5[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1152L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr5 + static_cast<long>(x1 + (128L*x2) + (1152L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)), static_cast<long>(256L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr6[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (2304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr6 + static_cast<long>(x1 + (256L*x2) + (2304L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(50176L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr7[static_cast<long>(x2 + (50176L*x1) + (150528L*x0))];
                        out_ptr7[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       long* out_ptr2)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(56L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<long>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = out_ptr0[static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp10 ? tmp11() : -std::numeric_limits<decltype(tmp11())>::infinity();
                            auto tmp14 = c10::convert<long>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = out_ptr0[static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp20;
                            }
                            ;
                            auto tmp21 = tmp18 ? tmp19() : -std::numeric_limits<decltype(tmp19())>::infinity();
                            auto tmp22 = max_propagate_nan(tmp21, tmp13);
                            auto tmp23 = c10::convert<long>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = out_ptr0[static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp29;
                            }
                            ;
                            auto tmp30 = tmp27 ? tmp28() : -std::numeric_limits<decltype(tmp28())>::infinity();
                            auto tmp31 = max_propagate_nan(tmp30, tmp22);
                            auto tmp32 = c10::convert<long>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = out_ptr0[static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp38;
                            }
                            ;
                            auto tmp39 = tmp36 ? tmp37() : -std::numeric_limits<decltype(tmp37())>::infinity();
                            auto tmp40 = max_propagate_nan(tmp39, tmp31);
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = out_ptr0[static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp41 ? tmp42() : -std::numeric_limits<decltype(tmp42())>::infinity();
                            auto tmp45 = max_propagate_nan(tmp44, tmp40);
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp48;
                            }
                            ;
                            auto tmp49 = tmp46 ? tmp47() : -std::numeric_limits<decltype(tmp47())>::infinity();
                            auto tmp50 = max_propagate_nan(tmp49, tmp45);
                            auto tmp51 = c10::convert<long>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp57;
                            }
                            ;
                            auto tmp58 = tmp55 ? tmp56() : -std::numeric_limits<decltype(tmp56())>::infinity();
                            auto tmp59 = max_propagate_nan(tmp58, tmp50);
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp62;
                            }
                            ;
                            auto tmp63 = tmp60 ? tmp61() : -std::numeric_limits<decltype(tmp61())>::infinity();
                            auto tmp64 = max_propagate_nan(tmp63, tmp59);
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = out_ptr0[static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp67;
                            }
                            ;
                            auto tmp68 = tmp65 ? tmp66() : -std::numeric_limits<decltype(tmp66())>::infinity();
                            auto tmp69 = max_propagate_nan(tmp68, tmp64);
                            auto tmp70 = [&]
                            {
                                auto tmp71 = out_ptr0[static_cast<long>((-7232L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp71;
                            }
                            ;
                            auto tmp72 = tmp10 ? tmp70() : -std::numeric_limits<decltype(tmp70())>::infinity();
                            auto tmp73 = [&]
                            {
                                auto tmp74 = out_ptr0[static_cast<long>((-7168L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp74;
                            }
                            ;
                            auto tmp75 = tmp18 ? tmp73() : -std::numeric_limits<decltype(tmp73())>::infinity();
                            auto tmp76 = tmp75 > tmp72;
                            auto tmp77 = c10::convert<long>((-112L) + (2L*x2) + (224L*x1));
                            auto tmp78 = c10::convert<long>((-113L) + (2L*x2) + (224L*x1));
                            auto tmp79 = tmp76 ? tmp77 : tmp78;
                            auto tmp80 = max_propagate_nan(tmp75, tmp72);
                            auto tmp81 = [&]
                            {
                                auto tmp82 = out_ptr0[static_cast<long>((-7104L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp82;
                            }
                            ;
                            auto tmp83 = tmp27 ? tmp81() : -std::numeric_limits<decltype(tmp81())>::infinity();
                            auto tmp84 = tmp83 > tmp80;
                            auto tmp85 = c10::convert<long>((-111L) + (2L*x2) + (224L*x1));
                            auto tmp86 = tmp84 ? tmp85 : tmp79;
                            auto tmp87 = max_propagate_nan(tmp83, tmp80);
                            auto tmp88 = [&]
                            {
                                auto tmp89 = out_ptr0[static_cast<long>((-64L) + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp89;
                            }
                            ;
                            auto tmp90 = tmp36 ? tmp88() : -std::numeric_limits<decltype(tmp88())>::infinity();
                            auto tmp91 = tmp90 > tmp87;
                            auto tmp92 = c10::convert<long>((-1L) + (2L*x2) + (224L*x1));
                            auto tmp93 = tmp91 ? tmp92 : tmp86;
                            auto tmp94 = max_propagate_nan(tmp90, tmp87);
                            auto tmp95 = [&]
                            {
                                auto tmp96 = out_ptr0[static_cast<long>(x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp96;
                            }
                            ;
                            auto tmp97 = tmp41 ? tmp95() : -std::numeric_limits<decltype(tmp95())>::infinity();
                            auto tmp98 = tmp97 > tmp94;
                            auto tmp99 = c10::convert<long>((2L*x2) + (224L*x1));
                            auto tmp100 = tmp98 ? tmp99 : tmp93;
                            auto tmp101 = max_propagate_nan(tmp97, tmp94);
                            auto tmp102 = [&]
                            {
                                auto tmp103 = out_ptr0[static_cast<long>(64L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp103;
                            }
                            ;
                            auto tmp104 = tmp46 ? tmp102() : -std::numeric_limits<decltype(tmp102())>::infinity();
                            auto tmp105 = tmp104 > tmp101;
                            auto tmp106 = c10::convert<long>(1L + (2L*x2) + (224L*x1));
                            auto tmp107 = tmp105 ? tmp106 : tmp100;
                            auto tmp108 = max_propagate_nan(tmp104, tmp101);
                            auto tmp109 = [&]
                            {
                                auto tmp110 = out_ptr0[static_cast<long>(7104L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp110;
                            }
                            ;
                            auto tmp111 = tmp55 ? tmp109() : -std::numeric_limits<decltype(tmp109())>::infinity();
                            auto tmp112 = tmp111 > tmp108;
                            auto tmp113 = c10::convert<long>(111L + (2L*x2) + (224L*x1));
                            auto tmp114 = tmp112 ? tmp113 : tmp107;
                            auto tmp115 = max_propagate_nan(tmp111, tmp108);
                            auto tmp116 = [&]
                            {
                                auto tmp117 = out_ptr0[static_cast<long>(7168L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp117;
                            }
                            ;
                            auto tmp118 = tmp60 ? tmp116() : -std::numeric_limits<decltype(tmp116())>::infinity();
                            auto tmp119 = tmp118 > tmp115;
                            auto tmp120 = c10::convert<long>(112L + (2L*x2) + (224L*x1));
                            auto tmp121 = tmp119 ? tmp120 : tmp114;
                            auto tmp122 = max_propagate_nan(tmp118, tmp115);
                            auto tmp123 = [&]
                            {
                                auto tmp124 = out_ptr0[static_cast<long>(7232L + x3 + (128L*x2) + (14336L*x1) + (802816L*x0))];
                                return tmp124;
                            }
                            ;
                            auto tmp125 = tmp65 ? tmp123() : -std::numeric_limits<decltype(tmp123())>::infinity();
                            auto tmp126 = tmp125 > tmp122;
                            auto tmp127 = c10::convert<long>(113L + (2L*x2) + (224L*x1));
                            auto tmp128 = tmp126 ? tmp127 : tmp121;
                            auto tmp129 = max_propagate_nan(tmp125, tmp122);
                            out_ptr1[static_cast<long>(x3 + (64L*x2) + (3584L*x1) + (200704L*x0))] = tmp69;
                            out_ptr2[static_cast<long>(x3 + (64L*x2) + (3584L*x1) + (200704L*x0))] = tmp128;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x1 + (128L*x2) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
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
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (32L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_mul_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (64L*x1) + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(64L + x2 + (128L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (64L*x1) + (128L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (128L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(64L + x1 + (128L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 / tmp3;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (2L*x1) + (2L*x1_inner) + (128L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((2L*x2) + (2L*x2_inner) + (128L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(64L + x2 + (128L*x1) + (401408L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>(1L + (2L*x2) + (2L*x2_inner) + (128L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 + tmp5;
                        tmp6.store(out_ptr2 + static_cast<long>(x2 + (64L*x1) + (200704L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp30.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3211264L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (802816L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x1 + (256L*x2) + (802816L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(3136.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
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
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (64L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (128L*x1) + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(128L + x2 + (256L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (128L*x1) + (256L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (128L*x2) + (256L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(128L + x1 + (256L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 / tmp3;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (2L*x1) + (2L*x1_inner) + (256L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(128L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (256L*x1) + (802816L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((2L*x2) + (2L*x2_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(128L + x2 + (256L*x1) + (802816L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>(1L + (2L*x2) + (2L*x2_inner) + (256L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 + tmp5;
                        tmp6.store(out_ptr2 + static_cast<long>(x2 + (128L*x1) + (401408L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(28L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(128L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-7296L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(out_ptr2 + static_cast<long>((-7168L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(out_ptr2 + static_cast<long>((-7040L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(out_ptr2 + static_cast<long>((-128L) + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr2 + static_cast<long>(x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr2 + static_cast<long>(128L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(out_ptr2 + static_cast<long>(7040L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr2 + static_cast<long>(7168L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr2 + static_cast<long>(7296L + x3 + (256L*x2) + (14336L*x1) + (401408L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(57);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(56);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(56);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(2L*x2);
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
                                auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(56);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + (2L*x2));
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
                                auto tmp145 = c10::convert<int>(2L*x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(56);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp164 = c10::convert<int>(2L*x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(56);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(2L*x2);
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
                                auto tmp183 = c10::convert<int>(2L*x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(56);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + (2L*x2));
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
                                auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(56);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(56);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(2L*x2);
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
                                auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(56);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + (2L*x2));
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
                            tmp261.store(out_ptr3 + static_cast<long>(x3 + (128L*x2) + (3584L*x1) + (100352L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(28L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14592L + x2 + (512L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_14 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp30.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1605632L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x1 + (512L*x2) + (401408L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (256L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1024L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(784.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
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
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (128L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (256L*x1) + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(256L + x2 + (512L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (256L*x1) + (512L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (256L*x2) + (512L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(256L + x1 + (512L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 / tmp3;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (2L*x1) + (2L*x1_inner) + (512L*x0))] = tmpbuf[x1_inner]; }
                }
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (512L*x1) + (401408L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((2L*x2) + (2L*x2_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(256L + x2 + (512L*x1) + (401408L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>(1L + (2L*x2) + (2L*x2_inner) + (512L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 + tmp5;
                        tmp6.store(out_ptr2 + static_cast<long>(x2 + (256L*x1) + (200704L*x0)));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(14L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                            auto tmp7 = tmp6 >= tmp1;
                            auto tmp8 = tmp6 < tmp3;
                            auto tmp9 = tmp7 & tmp8;
                            auto tmp10 = tmp5 & tmp9;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-7424L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            auto tmp14 = c10::convert<int>(2L*x2);
                            auto tmp15 = tmp14 >= tmp1;
                            auto tmp16 = tmp14 < tmp3;
                            auto tmp17 = tmp15 & tmp16;
                            auto tmp18 = tmp5 & tmp17;
                            auto tmp19 = [&]
                            {
                                auto tmp20 = masked_load(out_ptr2 + static_cast<long>((-7168L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp18));
                                return tmp20;
                            }
                            ;
                            auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                            auto tmp22 = tmp21 + tmp13;
                            auto tmp23 = c10::convert<int>(1L + (2L*x2));
                            auto tmp24 = tmp23 >= tmp1;
                            auto tmp25 = tmp23 < tmp3;
                            auto tmp26 = tmp24 & tmp25;
                            auto tmp27 = tmp5 & tmp26;
                            auto tmp28 = [&]
                            {
                                auto tmp29 = masked_load(out_ptr2 + static_cast<long>((-6912L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp27));
                                return tmp29;
                            }
                            ;
                            auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                            auto tmp31 = tmp30 + tmp22;
                            auto tmp32 = c10::convert<int>(2L*x1);
                            auto tmp33 = tmp32 >= tmp1;
                            auto tmp34 = tmp32 < tmp3;
                            auto tmp35 = tmp33 & tmp34;
                            auto tmp36 = tmp35 & tmp9;
                            auto tmp37 = [&]
                            {
                                auto tmp38 = masked_load(out_ptr2 + static_cast<long>((-256L) + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp36));
                                return tmp38;
                            }
                            ;
                            auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                            auto tmp40 = tmp39 + tmp31;
                            auto tmp41 = tmp35 & tmp17;
                            auto tmp42 = [&]
                            {
                                auto tmp43 = masked_load(out_ptr2 + static_cast<long>(x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp41));
                                return tmp43;
                            }
                            ;
                            auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                            auto tmp45 = tmp44 + tmp40;
                            auto tmp46 = tmp35 & tmp26;
                            auto tmp47 = [&]
                            {
                                auto tmp48 = masked_load(out_ptr2 + static_cast<long>(256L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp46));
                                return tmp48;
                            }
                            ;
                            auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                            auto tmp50 = tmp49 + tmp45;
                            auto tmp51 = c10::convert<int>(1L + (2L*x1));
                            auto tmp52 = tmp51 >= tmp1;
                            auto tmp53 = tmp51 < tmp3;
                            auto tmp54 = tmp52 & tmp53;
                            auto tmp55 = tmp54 & tmp9;
                            auto tmp56 = [&]
                            {
                                auto tmp57 = masked_load(out_ptr2 + static_cast<long>(6912L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp55));
                                return tmp57;
                            }
                            ;
                            auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                            auto tmp59 = tmp58 + tmp50;
                            auto tmp60 = tmp54 & tmp17;
                            auto tmp61 = [&]
                            {
                                auto tmp62 = masked_load(out_ptr2 + static_cast<long>(7168L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp60));
                                return tmp62;
                            }
                            ;
                            auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                            auto tmp64 = tmp63 + tmp59;
                            auto tmp65 = tmp54 & tmp26;
                            auto tmp66 = [&]
                            {
                                auto tmp67 = masked_load(out_ptr2 + static_cast<long>(7424L + x3 + (512L*x2) + (14336L*x1) + (200704L*x0)), to_float_mask(tmp65));
                                return tmp67;
                            }
                            ;
                            auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                            auto tmp69 = tmp68 + tmp64;
                            auto tmp70 = static_cast<int>(-1);
                            auto tmp71 = tmp0 >= tmp70;
                            auto tmp72 = static_cast<int>(29);
                            auto tmp73 = tmp0 < tmp72;
                            auto tmp74 = tmp71 & tmp73;
                            auto tmp75 = tmp6 >= tmp70;
                            auto tmp76 = tmp6 < tmp72;
                            auto tmp77 = tmp75 & tmp76;
                            auto tmp78 = tmp74 & tmp77;
                            auto tmp79 = [&]
                            {
                                auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp81 = static_cast<int>(0);
                                auto tmp82 = tmp80 >= tmp81;
                                auto tmp83 = static_cast<int>(28);
                                auto tmp84 = tmp80 < tmp83;
                                auto tmp85 = tmp82 & tmp84;
                                auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp102 = static_cast<int>(0);
                                auto tmp103 = tmp101 >= tmp102;
                                auto tmp104 = static_cast<int>(28);
                                auto tmp105 = tmp101 < tmp104;
                                auto tmp106 = tmp103 & tmp105;
                                auto tmp107 = c10::convert<int>(2L*x2);
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
                                auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp124 = static_cast<int>(0);
                                auto tmp125 = tmp123 >= tmp124;
                                auto tmp126 = static_cast<int>(28);
                                auto tmp127 = tmp123 < tmp126;
                                auto tmp128 = tmp125 & tmp127;
                                auto tmp129 = c10::convert<int>(1L + (2L*x2));
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
                                auto tmp145 = c10::convert<int>(2L*x1);
                                auto tmp146 = static_cast<int>(0);
                                auto tmp147 = tmp145 >= tmp146;
                                auto tmp148 = static_cast<int>(28);
                                auto tmp149 = tmp145 < tmp148;
                                auto tmp150 = tmp147 & tmp149;
                                auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp164 = c10::convert<int>(2L*x1);
                                auto tmp165 = static_cast<int>(0);
                                auto tmp166 = tmp164 >= tmp165;
                                auto tmp167 = static_cast<int>(28);
                                auto tmp168 = tmp164 < tmp167;
                                auto tmp169 = tmp166 & tmp168;
                                auto tmp170 = c10::convert<int>(2L*x2);
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
                                auto tmp183 = c10::convert<int>(2L*x1);
                                auto tmp184 = static_cast<int>(0);
                                auto tmp185 = tmp183 >= tmp184;
                                auto tmp186 = static_cast<int>(28);
                                auto tmp187 = tmp183 < tmp186;
                                auto tmp188 = tmp185 & tmp187;
                                auto tmp189 = c10::convert<int>(1L + (2L*x2));
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
                                auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                auto tmp206 = static_cast<int>(0);
                                auto tmp207 = tmp205 >= tmp206;
                                auto tmp208 = static_cast<int>(28);
                                auto tmp209 = tmp205 < tmp208;
                                auto tmp210 = tmp207 & tmp209;
                                auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
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
                                auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                auto tmp225 = static_cast<int>(0);
                                auto tmp226 = tmp224 >= tmp225;
                                auto tmp227 = static_cast<int>(28);
                                auto tmp228 = tmp224 < tmp227;
                                auto tmp229 = tmp226 & tmp228;
                                auto tmp230 = c10::convert<int>(2L*x2);
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
                                auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                auto tmp244 = static_cast<int>(0);
                                auto tmp245 = tmp243 >= tmp244;
                                auto tmp246 = static_cast<int>(28);
                                auto tmp247 = tmp243 < tmp246;
                                auto tmp248 = tmp245 & tmp247;
                                auto tmp249 = c10::convert<int>(1L + (2L*x2));
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
                            tmp261.store(out_ptr3 + static_cast<long>(x3 + (256L*x2) + (3584L*x1) + (50176L*x0)));
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
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(56L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(14L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14848L + x2 + (1024L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_20 = async_compile.cpp('''
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
                       const float* in_ptr9)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp30.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(802816L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (512L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (512L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1024L*x0)));
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
                    tmp17.store(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x2) + (200704L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x1 + (1024L*x2) + (200704L*x0)));
                            auto tmp2 = tmp0 + tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (512L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (256L*x0)));
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
                tmp17.store(out_ptr0 + static_cast<long>(x1 + (256L*x0)));
            }
        }
    }
}
''')


cpp_fused__softmax_avg_pool2d_mul_sum_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (512L*x1) + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(512L + x2 + (1024L*x0)));
                    auto tmp3 = at::vec::maximum(tmp1, tmp2);
                    auto tmp4 = tmp0 - tmp3;
                    auto tmp5 = tmp4.exp();
                    tmp5.store(out_ptr0 + static_cast<long>(x2 + (512L*x1) + (1024L*x0)));
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(8L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (512L*x2) + (1024L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1024L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(512L + x1 + (1024L*x0)));
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = tmp0 / tmp3;
                    { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp4.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (2L*x1) + (2L*x1_inner) + (1024L*x0))] = tmpbuf[x1_inner]; }
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(512L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (1024L*x1) + (200704L*x0)));
                        auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((2L*x2) + (2L*x2_inner) + (1024L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(512L + x2 + (1024L*x1) + (200704L*x0)));
                        auto tmp4 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>(1L + (2L*x2) + (2L*x2_inner) + (1024L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp5 = tmp3 * tmp4;
                        auto tmp6 = tmp2 + tmp5;
                        tmp6.store(out_ptr2 + static_cast<long>(x2 + (512L*x1) + (100352L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                #pragma GCC ivdep
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(7L); x2+=static_cast<long>(1L))
                        {
                            for(long x3=static_cast<long>(0L); x3<static_cast<long>(512L); x3+=static_cast<long>(8L))
                            {
                                auto tmp0 = c10::convert<int>((-1L) + (2L*x1));
                                auto tmp1 = static_cast<int>(0);
                                auto tmp2 = tmp0 >= tmp1;
                                auto tmp3 = static_cast<int>(14);
                                auto tmp4 = tmp0 < tmp3;
                                auto tmp5 = tmp2 & tmp4;
                                auto tmp6 = c10::convert<int>((-1L) + (2L*x2));
                                auto tmp7 = tmp6 >= tmp1;
                                auto tmp8 = tmp6 < tmp3;
                                auto tmp9 = tmp7 & tmp8;
                                auto tmp10 = tmp5 & tmp9;
                                auto tmp11 = [&]
                                {
                                    auto tmp12 = masked_load(out_ptr2 + static_cast<long>((-7680L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp10));
                                    return tmp12;
                                }
                                ;
                                auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                                auto tmp14 = c10::convert<int>(2L*x2);
                                auto tmp15 = tmp14 >= tmp1;
                                auto tmp16 = tmp14 < tmp3;
                                auto tmp17 = tmp15 & tmp16;
                                auto tmp18 = tmp5 & tmp17;
                                auto tmp19 = [&]
                                {
                                    auto tmp20 = masked_load(out_ptr2 + static_cast<long>((-7168L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp18));
                                    return tmp20;
                                }
                                ;
                                auto tmp21 = decltype(tmp19())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp19(), to_float_mask(tmp18));
                                auto tmp22 = tmp21 + tmp13;
                                auto tmp23 = c10::convert<int>(1L + (2L*x2));
                                auto tmp24 = tmp23 >= tmp1;
                                auto tmp25 = tmp23 < tmp3;
                                auto tmp26 = tmp24 & tmp25;
                                auto tmp27 = tmp5 & tmp26;
                                auto tmp28 = [&]
                                {
                                    auto tmp29 = masked_load(out_ptr2 + static_cast<long>((-6656L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp27));
                                    return tmp29;
                                }
                                ;
                                auto tmp30 = decltype(tmp28())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp28(), to_float_mask(tmp27));
                                auto tmp31 = tmp30 + tmp22;
                                auto tmp32 = c10::convert<int>(2L*x1);
                                auto tmp33 = tmp32 >= tmp1;
                                auto tmp34 = tmp32 < tmp3;
                                auto tmp35 = tmp33 & tmp34;
                                auto tmp36 = tmp35 & tmp9;
                                auto tmp37 = [&]
                                {
                                    auto tmp38 = masked_load(out_ptr2 + static_cast<long>((-512L) + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp36));
                                    return tmp38;
                                }
                                ;
                                auto tmp39 = decltype(tmp37())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp37(), to_float_mask(tmp36));
                                auto tmp40 = tmp39 + tmp31;
                                auto tmp41 = tmp35 & tmp17;
                                auto tmp42 = [&]
                                {
                                    auto tmp43 = masked_load(out_ptr2 + static_cast<long>(x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp41));
                                    return tmp43;
                                }
                                ;
                                auto tmp44 = decltype(tmp42())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp42(), to_float_mask(tmp41));
                                auto tmp45 = tmp44 + tmp40;
                                auto tmp46 = tmp35 & tmp26;
                                auto tmp47 = [&]
                                {
                                    auto tmp48 = masked_load(out_ptr2 + static_cast<long>(512L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp46));
                                    return tmp48;
                                }
                                ;
                                auto tmp49 = decltype(tmp47())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp47(), to_float_mask(tmp46));
                                auto tmp50 = tmp49 + tmp45;
                                auto tmp51 = c10::convert<int>(1L + (2L*x1));
                                auto tmp52 = tmp51 >= tmp1;
                                auto tmp53 = tmp51 < tmp3;
                                auto tmp54 = tmp52 & tmp53;
                                auto tmp55 = tmp54 & tmp9;
                                auto tmp56 = [&]
                                {
                                    auto tmp57 = masked_load(out_ptr2 + static_cast<long>(6656L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp55));
                                    return tmp57;
                                }
                                ;
                                auto tmp58 = decltype(tmp56())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp56(), to_float_mask(tmp55));
                                auto tmp59 = tmp58 + tmp50;
                                auto tmp60 = tmp54 & tmp17;
                                auto tmp61 = [&]
                                {
                                    auto tmp62 = masked_load(out_ptr2 + static_cast<long>(7168L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp60));
                                    return tmp62;
                                }
                                ;
                                auto tmp63 = decltype(tmp61())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp61(), to_float_mask(tmp60));
                                auto tmp64 = tmp63 + tmp59;
                                auto tmp65 = tmp54 & tmp26;
                                auto tmp66 = [&]
                                {
                                    auto tmp67 = masked_load(out_ptr2 + static_cast<long>(7680L + x3 + (1024L*x2) + (14336L*x1) + (100352L*x0)), to_float_mask(tmp65));
                                    return tmp67;
                                }
                                ;
                                auto tmp68 = decltype(tmp66())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp66(), to_float_mask(tmp65));
                                auto tmp69 = tmp68 + tmp64;
                                auto tmp70 = static_cast<int>(-1);
                                auto tmp71 = tmp0 >= tmp70;
                                auto tmp72 = static_cast<int>(15);
                                auto tmp73 = tmp0 < tmp72;
                                auto tmp74 = tmp71 & tmp73;
                                auto tmp75 = tmp6 >= tmp70;
                                auto tmp76 = tmp6 < tmp72;
                                auto tmp77 = tmp75 & tmp76;
                                auto tmp78 = tmp74 & tmp77;
                                auto tmp79 = [&]
                                {
                                    auto tmp80 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp81 = static_cast<int>(0);
                                    auto tmp82 = tmp80 >= tmp81;
                                    auto tmp83 = static_cast<int>(14);
                                    auto tmp84 = tmp80 < tmp83;
                                    auto tmp85 = tmp82 & tmp84;
                                    auto tmp86 = c10::convert<int>((-1L) + (2L*x2));
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
                                    auto tmp101 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp102 = static_cast<int>(0);
                                    auto tmp103 = tmp101 >= tmp102;
                                    auto tmp104 = static_cast<int>(14);
                                    auto tmp105 = tmp101 < tmp104;
                                    auto tmp106 = tmp103 & tmp105;
                                    auto tmp107 = c10::convert<int>(2L*x2);
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
                                    auto tmp123 = c10::convert<int>((-1L) + (2L*x1));
                                    auto tmp124 = static_cast<int>(0);
                                    auto tmp125 = tmp123 >= tmp124;
                                    auto tmp126 = static_cast<int>(14);
                                    auto tmp127 = tmp123 < tmp126;
                                    auto tmp128 = tmp125 & tmp127;
                                    auto tmp129 = c10::convert<int>(1L + (2L*x2));
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
                                    auto tmp145 = c10::convert<int>(2L*x1);
                                    auto tmp146 = static_cast<int>(0);
                                    auto tmp147 = tmp145 >= tmp146;
                                    auto tmp148 = static_cast<int>(14);
                                    auto tmp149 = tmp145 < tmp148;
                                    auto tmp150 = tmp147 & tmp149;
                                    auto tmp151 = c10::convert<int>((-1L) + (2L*x2));
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
                                    auto tmp164 = c10::convert<int>(2L*x1);
                                    auto tmp165 = static_cast<int>(0);
                                    auto tmp166 = tmp164 >= tmp165;
                                    auto tmp167 = static_cast<int>(14);
                                    auto tmp168 = tmp164 < tmp167;
                                    auto tmp169 = tmp166 & tmp168;
                                    auto tmp170 = c10::convert<int>(2L*x2);
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
                                    auto tmp183 = c10::convert<int>(2L*x1);
                                    auto tmp184 = static_cast<int>(0);
                                    auto tmp185 = tmp183 >= tmp184;
                                    auto tmp186 = static_cast<int>(14);
                                    auto tmp187 = tmp183 < tmp186;
                                    auto tmp188 = tmp185 & tmp187;
                                    auto tmp189 = c10::convert<int>(1L + (2L*x2));
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
                                    auto tmp205 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp206 = static_cast<int>(0);
                                    auto tmp207 = tmp205 >= tmp206;
                                    auto tmp208 = static_cast<int>(14);
                                    auto tmp209 = tmp205 < tmp208;
                                    auto tmp210 = tmp207 & tmp209;
                                    auto tmp211 = c10::convert<int>((-1L) + (2L*x2));
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
                                    auto tmp224 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp225 = static_cast<int>(0);
                                    auto tmp226 = tmp224 >= tmp225;
                                    auto tmp227 = static_cast<int>(14);
                                    auto tmp228 = tmp224 < tmp227;
                                    auto tmp229 = tmp226 & tmp228;
                                    auto tmp230 = c10::convert<int>(2L*x2);
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
                                    auto tmp243 = c10::convert<int>(1L + (2L*x1));
                                    auto tmp244 = static_cast<int>(0);
                                    auto tmp245 = tmp243 >= tmp244;
                                    auto tmp246 = static_cast<int>(14);
                                    auto tmp247 = tmp243 < tmp246;
                                    auto tmp248 = tmp245 & tmp247;
                                    auto tmp249 = c10::convert<int>(1L + (2L*x2));
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
                                tmp261.store(out_ptr3 + static_cast<long>(x3 + (512L*x2) + (3584L*x1) + (25088L*x0)));
                            }
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_avg_pool2d_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(28L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(7L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1024L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(14336L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(15360L + x2 + (2048L*x1) + (28672L*x0)));
                        auto tmp2 = tmp1 + tmp0;
                        auto tmp4 = tmp3 + tmp2;
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = static_cast<float>(0.25);
                        auto tmp8 = at::vec::Vectorized<float>(tmp7);
                        auto tmp9 = tmp6 * tmp8;
                        tmp9.store(out_ptr0 + static_cast<long>(x2 + (1024L*x1) + (7168L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_view_26 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (2048L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
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
                    tmp30.store(out_ptr0 + static_cast<long>(x1 + (2048L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (2048L*x2) + (100352L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (2048L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_relu_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       bool* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(401408L); x0+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(x0)];
                auto tmp1 = tmp0 * (tmp0>0);
                auto tmp2 = static_cast<float>(0.0);
                auto tmp3 = tmp1 <= tmp2;
                out_ptr0[static_cast<long>(x0)] = tmp3;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (64, ), (1, ))
    assert_size_stride(primals_36, (64, ), (1, ))
    assert_size_stride(primals_37, (64, ), (1, ))
    assert_size_stride(primals_38, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_50, (512, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_57, (512, ), (1, ))
    assert_size_stride(primals_58, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_61, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (1024, ), (1, ))
    assert_size_stride(primals_63, (1024, ), (1, ))
    assert_size_stride(primals_64, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_69, (1024, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (1024, ), (1, ))
    assert_size_stride(primals_76, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_80, (2048, ), (1, ))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_82, (1000, 2048), (2048, 1))
    assert_size_stride(primals_83, (1000, ), (1, ))
    assert_size_stride(primals_84, (32, ), (1, ))
    assert_size_stride(primals_85, (32, ), (1, ))
    assert_size_stride(primals_86, (), ())
    assert_size_stride(primals_87, (32, ), (1, ))
    assert_size_stride(primals_88, (32, ), (1, ))
    assert_size_stride(primals_89, (), ())
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, ), (1, ))
    assert_size_stride(primals_92, (), ())
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (), ())
    assert_size_stride(primals_96, (128, ), (1, ))
    assert_size_stride(primals_97, (128, ), (1, ))
    assert_size_stride(primals_98, (), ())
    assert_size_stride(primals_99, (32, ), (1, ))
    assert_size_stride(primals_100, (32, ), (1, ))
    assert_size_stride(primals_101, (), ())
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (256, ), (1, ))
    assert_size_stride(primals_104, (), ())
    assert_size_stride(primals_105, (256, ), (1, ))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (), ())
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (128, ), (1, ))
    assert_size_stride(primals_110, (), ())
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (), ())
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (), ())
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (512, ), (1, ))
    assert_size_stride(primals_119, (), ())
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (), ())
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (), ())
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (), ())
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, ), (1, ))
    assert_size_stride(primals_134, (), ())
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, ), (1, ))
    assert_size_stride(primals_137, (), ())
    assert_size_stride(primals_138, (512, ), (1, ))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_140, (), ())
    assert_size_stride(primals_141, (1024, ), (1, ))
    assert_size_stride(primals_142, (1024, ), (1, ))
    assert_size_stride(primals_143, (), ())
    assert_size_stride(primals_144, (256, ), (1, ))
    assert_size_stride(primals_145, (256, ), (1, ))
    assert_size_stride(primals_146, (), ())
    assert_size_stride(primals_147, (2048, ), (1, ))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_149, (), ())
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, ), (1, ))
    assert_size_stride(primals_152, (), ())
    assert_size_stride(primals_153, (4, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((512, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((1024, 256, 3, 3), (2304, 1, 768, 256), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_0(c_void_p(primals_1.data_ptr()), c_void_p(primals_4.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()))
    del primals_1
    del primals_13
    del primals_153
    del primals_31
    del primals_4
    del primals_49
    del primals_67
    del primals_7
    # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
    buf8 = extern_kernels.convolution(buf7, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf8, (4, 32, 112, 112), (401408, 1, 3584, 32))
    buf9 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf8.data_ptr()), c_void_p(primals_84.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(primals_2.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf9.data_ptr()))
    del primals_3
    # Source Nodes: [l__mod___conv1_3], Original ATen: [aten.convolution]
    buf10 = extern_kernels.convolution(buf9, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf10, (4, 32, 112, 112), (401408, 1, 3584, 32))
    buf11 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf10.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(primals_88.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(primals_6.data_ptr()), c_void_p(buf11.data_ptr()))
    del primals_6
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf12 = extern_kernels.convolution(buf11, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (4, 64, 112, 112), (802816, 1, 7168, 64))
    buf13 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    buf15 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.int64)
    cpp_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3(c_void_p(buf12.data_ptr()), c_void_p(primals_90.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()))
    del primals_9
    # Source Nodes: [out], Original ATen: [aten.convolution]
    buf16 = extern_kernels.convolution(buf14, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf16, (4, 64, 56, 56), (200704, 1, 3584, 64))
    buf17 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_4(c_void_p(buf16.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(primals_94.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(primals_12.data_ptr()), c_void_p(buf17.data_ptr()))
    del primals_12
    # Source Nodes: [x_4], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(buf17, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf18, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf19 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf20 = empty_strided((4, 64, 1, 1), (64, 1, 256, 256), device='cpu', dtype=torch.float32)
    buf21 = reinterpret_tensor(buf20, (4, 64, 1, 1), (64, 1, 64, 64), 0); del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_5(c_void_p(buf21.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(primals_96.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf19.data_ptr()))
    del primals_15
    # Source Nodes: [x_gap_2], Original ATen: [aten.convolution]
    buf22 = extern_kernels.convolution(buf21, primals_16, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf22, (4, 32, 1, 1), (32, 1, 32, 32))
    del primals_17
    buf23 = empty_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_6(c_void_p(buf22.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(primals_100.data_ptr()), c_void_p(primals_18.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf23.data_ptr()))
    del primals_19
    # Source Nodes: [x_attn], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf23, primals_20, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf24, (4, 128, 1, 1), (128, 1, 128, 128))
    del primals_21
    buf25 = empty_strided((4, 2, 1, 64), (128, 64, 512, 1), device='cpu', dtype=torch.float32)
    buf26 = empty_strided((4, 2, 1, 64), (128, 1, 128, 2), device='cpu', dtype=torch.float32)
    buf27 = empty_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_mul_sum_7(c_void_p(buf24.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(buf27.data_ptr()))
    # Source Nodes: [out_8], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(buf27, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (4, 256, 56, 56), (802816, 1, 14336, 256))
    # Source Nodes: [getattr_l__mod___layer1___0___downsample_1], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf14, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (4, 256, 56, 56), (802816, 1, 14336, 256))
    buf30 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf31 = buf30; del buf30  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_8(c_void_p(buf31.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(primals_102.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_24.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(primals_106.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(primals_27.data_ptr()))
    del primals_24
    del primals_27
    # Source Nodes: [out_12], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (4, 128, 56, 56), (401408, 1, 7168, 128))
    buf33 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf32.data_ptr()), c_void_p(primals_108.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(primals_30.data_ptr()), c_void_p(buf33.data_ptr()))
    del primals_30
    # Source Nodes: [x_13], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf34, (4, 256, 56, 56), (802816, 1, 14336, 256))
    buf35 = empty_strided((4, 256, 56, 56), (802816, 1, 14336, 256), device='cpu', dtype=torch.float32)
    buf36 = reinterpret_tensor(buf25, (4, 128, 1, 1), (128, 1, 512, 512), 0); del buf25  # reuse
    buf37 = reinterpret_tensor(buf36, (4, 128, 1, 1), (128, 1, 128, 128), 0); del buf36  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_10(c_void_p(buf37.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(primals_112.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf35.data_ptr()))
    del primals_33
    # Source Nodes: [x_gap_7], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, primals_34, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf38, (4, 64, 1, 1), (64, 1, 64, 64))
    del primals_35
    buf39 = empty_strided((4, 64, 1, 1), (64, 1, 64, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf38.data_ptr()), c_void_p(primals_114.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(primals_36.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf39.data_ptr()))
    del primals_37
    # Source Nodes: [x_attn_2], Original ATen: [aten.convolution]
    buf40 = extern_kernels.convolution(buf39, primals_38, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf40, (4, 256, 1, 1), (256, 1, 256, 256))
    del primals_39
    buf41 = empty_strided((4, 2, 1, 128), (256, 128, 1024, 1), device='cpu', dtype=torch.float32)
    buf42 = empty_strided((4, 2, 1, 128), (256, 1, 256, 2), device='cpu', dtype=torch.float32)
    buf43 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cpu', dtype=torch.float32)
    buf44 = empty_strided((4, 128, 28, 28), (100352, 1, 3584, 128), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_12(c_void_p(buf40.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()))
    # Source Nodes: [out_21], Original ATen: [aten.convolution]
    buf45 = extern_kernels.convolution(buf44, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf45, (4, 512, 28, 28), (401408, 1, 14336, 512))
    buf46 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_13(c_void_p(buf31.data_ptr()), c_void_p(buf46.data_ptr()))
    # Source Nodes: [getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.convolution]
    buf47 = extern_kernels.convolution(buf46, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (4, 512, 28, 28), (401408, 1, 14336, 512))
    buf48 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_14(c_void_p(buf49.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(primals_118.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(primals_44.data_ptr()), c_void_p(primals_45.data_ptr()))
    del primals_42
    del primals_45
    # Source Nodes: [out_25], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(buf49, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (4, 256, 28, 28), (200704, 1, 7168, 256))
    buf51 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_15(c_void_p(buf50.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_124.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(primals_48.data_ptr()), c_void_p(buf51.data_ptr()))
    del primals_48
    # Source Nodes: [x_22], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf51, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf52, (4, 512, 28, 28), (401408, 1, 14336, 512))
    buf53 = empty_strided((4, 512, 28, 28), (401408, 1, 14336, 512), device='cpu', dtype=torch.float32)
    buf54 = reinterpret_tensor(buf41, (4, 256, 1, 1), (256, 1, 1024, 1024), 0); del buf41  # reuse
    buf55 = reinterpret_tensor(buf54, (4, 256, 1, 1), (256, 1, 256, 256), 0); del buf54  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_16(c_void_p(buf55.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(primals_126.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(primals_50.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf53.data_ptr()))
    del primals_51
    # Source Nodes: [x_gap_12], Original ATen: [aten.convolution]
    buf56 = extern_kernels.convolution(buf55, primals_52, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf56, (4, 128, 1, 1), (128, 1, 128, 128))
    del primals_53
    buf57 = buf24; del buf24  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_17(c_void_p(buf56.data_ptr()), c_void_p(primals_129.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(primals_54.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf57.data_ptr()))
    del primals_55
    # Source Nodes: [x_attn_4], Original ATen: [aten.convolution]
    buf58 = extern_kernels.convolution(buf57, primals_56, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf58, (4, 512, 1, 1), (512, 1, 512, 512))
    del primals_57
    buf59 = empty_strided((4, 2, 1, 256), (512, 256, 2048, 1), device='cpu', dtype=torch.float32)
    buf60 = empty_strided((4, 2, 1, 256), (512, 1, 512, 2), device='cpu', dtype=torch.float32)
    buf61 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cpu', dtype=torch.float32)
    buf62 = empty_strided((4, 256, 14, 14), (50176, 1, 3584, 256), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_18(c_void_p(buf58.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del buf58
    # Source Nodes: [out_34], Original ATen: [aten.convolution]
    buf63 = extern_kernels.convolution(buf62, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf63, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    buf64 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_19(c_void_p(buf49.data_ptr()), c_void_p(buf64.data_ptr()))
    # Source Nodes: [getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.convolution]
    buf65 = extern_kernels.convolution(buf64, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf65, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    buf66 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    buf67 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_20(c_void_p(buf67.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(primals_132.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(primals_60.data_ptr()), c_void_p(buf65.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(primals_136.data_ptr()), c_void_p(primals_62.data_ptr()), c_void_p(primals_63.data_ptr()))
    del primals_60
    del primals_63
    # Source Nodes: [out_38], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (4, 512, 14, 14), (100352, 1, 7168, 512))
    buf69 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_relu_21(c_void_p(buf68.data_ptr()), c_void_p(primals_138.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(primals_66.data_ptr()), c_void_p(buf69.data_ptr()))
    del primals_66
    # Source Nodes: [x_31], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(buf69, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
    assert_size_stride(buf70, (4, 1024, 14, 14), (200704, 1, 14336, 1024))
    buf71 = empty_strided((4, 1024, 14, 14), (200704, 1, 14336, 1024), device='cpu', dtype=torch.float32)
    buf72 = reinterpret_tensor(buf59, (4, 512, 1, 1), (512, 1, 2048, 2048), 0); del buf59  # reuse
    buf73 = reinterpret_tensor(buf72, (4, 512, 1, 1), (512, 1, 512, 512), 0); del buf72  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_sum_22(c_void_p(buf73.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(primals_141.data_ptr()), c_void_p(primals_142.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf71.data_ptr()))
    del primals_69
    # Source Nodes: [x_gap_17], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf73, primals_70, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf74, (4, 256, 1, 1), (256, 1, 256, 256))
    del primals_71
    buf75 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_23(c_void_p(buf74.data_ptr()), c_void_p(primals_144.data_ptr()), c_void_p(primals_145.data_ptr()), c_void_p(primals_72.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf75.data_ptr()))
    del primals_73
    # Source Nodes: [x_attn_6], Original ATen: [aten.convolution]
    buf76 = extern_kernels.convolution(buf75, primals_74, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf76, (4, 1024, 1, 1), (1024, 1, 1024, 1024))
    del primals_75
    buf77 = empty_strided((4, 2, 1, 512), (1024, 512, 4096, 1), device='cpu', dtype=torch.float32)
    buf78 = empty_strided((4, 2, 1, 512), (1024, 1, 1024, 2), device='cpu', dtype=torch.float32)
    buf79 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cpu', dtype=torch.float32)
    buf80 = empty_strided((4, 512, 7, 7), (25088, 1, 3584, 512), device='cpu', dtype=torch.float32)
    cpp_fused__softmax_avg_pool2d_mul_sum_24(c_void_p(buf76.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(buf80.data_ptr()))
    del buf76
    del buf77
    # Source Nodes: [out_47], Original ATen: [aten.convolution]
    buf81 = extern_kernels.convolution(buf80, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf81, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    buf82 = empty_strided((4, 1024, 7, 7), (50176, 1, 7168, 1024), device='cpu', dtype=torch.float32)
    cpp_fused_avg_pool2d_25(c_void_p(buf67.data_ptr()), c_void_p(buf82.data_ptr()))
    # Source Nodes: [getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.convolution]
    buf83 = extern_kernels.convolution(buf82, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf83, (4, 2048, 7, 7), (100352, 1, 14336, 2048))
    buf84 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.float32)
    buf85 = empty_strided((4, 2048, 1, 1), (2048, 1, 8192, 8192), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf85, (4, 2048), (2048, 1), 0); del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_view_26(c_void_p(buf86.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(primals_148.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(primals_78.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(primals_150.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf84.data_ptr()))
    del primals_78
    del primals_81
    buf87 = empty((4, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [pred], Original ATen: [aten.addmm]
    extern_kernels.addmm(primals_83, buf86, reinterpret_tensor(primals_82, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf87)
    del primals_83
    buf88 = empty_strided((4, 2048, 7, 7), (100352, 1, 14336, 2048), device='cpu', dtype=torch.bool)
    cpp_fused_relu_threshold_backward_27(c_void_p(buf84.data_ptr()), c_void_p(buf88.data_ptr()))
    return (buf87, buf0, primals_2, buf1, primals_5, buf2, primals_8, primals_10, primals_11, buf3, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, buf4, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, buf5, primals_50, primals_52, primals_54, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, buf6, primals_68, primals_70, primals_72, primals_74, primals_76, primals_77, primals_79, primals_80, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf21, buf22, buf23, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf37, buf38, buf39, buf42, buf43, buf44, buf45, buf46, buf47, buf49, buf50, buf51, buf52, buf53, buf55, buf56, buf57, buf60, buf61, buf62, buf63, buf64, buf65, buf67, buf68, buf69, buf70, buf71, buf73, buf74, buf75, buf78, buf79, buf80, buf81, buf82, buf83, buf86, reinterpret_tensor(primals_82, (1000, 2048), (2048, 1), 0), buf88, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_50 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_56 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_78 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((1000, 2048), (2048, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    primals_84 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_87 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_90 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_93 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_96 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_99 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_102 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_105 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_108 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_110 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_111 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_114 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_117 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_120 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_123 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_126 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_129 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_132 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_135 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_138 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_141 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((1024, ), (1, ), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_144 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((256, ), (1, ), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_147 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_150 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((2048, ), (1, ), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((), (), device='cpu', dtype=torch.int64)
    primals_153 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_resnest', benchmark_compiled_module)
