
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


cpp_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
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
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_convolution_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(8L))
                    {
                        float tmp19[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (401408L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2));
                            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                            auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2 + (32L*x1) + (32L*x1_inner) + (401408L*x0)));
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
                            tmp18.store(tmp19 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp19, 8, out_ptr0 + static_cast<long>(x1 + (12544L*x2) + (401408L*x0)), static_cast<long>(12544L));
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (12544L*x1) + (12544L*x1_inner) + (401408L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (16L*x2) + (200704L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_4 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(12544L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(200704L + x2 + (12544L*x1) + (12544L*x1_inner) + (401408L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (16L*x2) + (200704L*x0)), static_cast<long>(16L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(96);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (96L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(192);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-96L) + x1 + (96L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = tmp28 * (tmp28>0);
                    out_ptr0[static_cast<long>(x1 + (192L*x0))] = tmp29;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(64);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (64L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(128);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-64L) + x1 + (64L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(192);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-128L) + x1 + (64L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    auto tmp37 = tmp36 * (tmp36>0);
                    out_ptr0[static_cast<long>(x1 + (192L*x0))] = tmp37;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-20L) + x1 + (20L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (40L*x0))] = tmp28;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_relu_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(60);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (60L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(120);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-60L) + x1 + (60L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp29 = tmp28 * (tmp28>0);
                    out_ptr0[static_cast<long>(x1 + (120L*x0))] = tmp29;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (40L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(20);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (20L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(40);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-20L) + x1 + (20L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (40L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (240L*x2) + (752640L*x0)), static_cast<long>(240L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (240L*x2) + (752640L*x0)), static_cast<long>(240L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (752640L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (3136L*x1) + (3136L*x1_inner) + (752640L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (60L*x2) + (188160L*x0)), static_cast<long>(60L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (3136L*x1) + (752640L*x0))];
                        out_ptr1[static_cast<long>(x1 + (60L*x2) + (188160L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_12 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(188160L + x2 + (3136L*x1) + (3136L*x1_inner) + (752640L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (60L*x2) + (188160L*x0)), static_cast<long>(60L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(188160L + x2 + (3136L*x1) + (752640L*x0))];
                        out_ptr0[static_cast<long>(x1 + (60L*x2) + (188160L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_13 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(376320L + x2 + (3136L*x1) + (3136L*x1_inner) + (752640L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (60L*x2) + (188160L*x0)), static_cast<long>(60L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(376320L + x2 + (3136L*x1) + (752640L*x0))];
                        out_ptr0[static_cast<long>(x1 + (60L*x2) + (188160L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(564480L + x2 + (3136L*x1) + (3136L*x1_inner) + (752640L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (60L*x2) + (188160L*x0)), static_cast<long>(60L));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(56L); x1<static_cast<long>(60L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(564480L + x2 + (3136L*x1) + (752640L*x0))];
                        out_ptr0[static_cast<long>(x1 + (60L*x2) + (188160L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_15 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(60);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (60L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(120);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-60L) + x1 + (60L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(180);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-120L) + x1 + (60L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(240);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-180L) + x1 + (60L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (240L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (188160L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (240L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (240L*x1) + (188160L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_19 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_20 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (336L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_23 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (336L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_24 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(28);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (28L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(56);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (56L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_26 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_27 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (336L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_30 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (336L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_31 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(28);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (28L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(56);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (56L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_33 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_34 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(168);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (168L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(336);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-168L) + x1 + (168L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (336L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (336L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(168L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(131712L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (168L*x2) + (131712L*x0)), static_cast<long>(168L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(56L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (56L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(28);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (28L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(56);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-28L) + x1 + (28L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (56L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (336L*x2) + (263424L*x0)), static_cast<long>(336L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (112L*x2) + (87808L*x0)), static_cast<long>(112L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(87808L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (87808L*x0)), static_cast<long>(112L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(175616L + x2 + (784L*x1) + (784L*x1_inner) + (263424L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (112L*x2) + (87808L*x0)), static_cast<long>(112L));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(1L))
                {
                    auto tmp23 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp35 = in_ptr6[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(112);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (112L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(224);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-112L) + x1 + (112L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(336);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = [&]
                    {
                        auto tmp19 = in_ptr2[static_cast<long>((-224L) + x1 + (112L*x0))];
                        return tmp19;
                    }
                    ;
                    auto tmp20 = tmp15 ? tmp18() : static_cast<decltype(tmp18())>(0.0);
                    auto tmp21 = tmp11 ? tmp14 : tmp20;
                    auto tmp22 = tmp4 ? tmp7 : tmp21;
                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp26 = static_cast<float>(1e-05);
                    auto tmp27 = decltype(tmp25)(tmp25 + tmp26);
                    auto tmp28 = std::sqrt(tmp27);
                    auto tmp29 = 1 / tmp28;
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = decltype(tmp24)(tmp24 * tmp31);
                    auto tmp34 = decltype(tmp32)(tmp32 * tmp33);
                    auto tmp36 = decltype(tmp34)(tmp34 + tmp35);
                    out_ptr0[static_cast<long>(x1 + (336L*x0))] = tmp36;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (336L*x2) + (65856L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (336L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(336L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (336L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (336L*x1) + (65856L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_46 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (104L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_47 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(312);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (312L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(624);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-312L) + x1 + (312L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (122304L*x0))];
                        out_ptr2[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_48 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_49 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_50 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_51 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(156);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (156L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(312);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-156L) + x1 + (156L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(468);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-312L) + x1 + (156L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(624);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-468L) + x1 + (156L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (624L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_53 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (624L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (624L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_54 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(52);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (52L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(104);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (104L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_56 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(312);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (312L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(624);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-312L) + x1 + (312L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (122304L*x0))];
                        out_ptr2[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_57 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_58 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_60 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(156);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (156L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(312);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-156L) + x1 + (156L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(468);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-312L) + x1 + (156L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(624);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-468L) + x1 + (156L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (624L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_62 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (624L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (624L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(52);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (52L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(104);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (104L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_65 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(312);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (312L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(624);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-312L) + x1 + (312L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr1[static_cast<long>(x2 + (196L*x1) + (122304L*x0))];
                        out_ptr2[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(30576L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_67 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_68 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(152L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)), static_cast<long>(156L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (156L*x2) + (30576L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(152L); x1<static_cast<long>(156L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(91728L + x2 + (196L*x1) + (122304L*x0))];
                        out_ptr0[static_cast<long>(x1 + (156L*x2) + (30576L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_69 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(156);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (156L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(312);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-156L) + x1 + (156L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(468);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-312L) + x1 + (156L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(624);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-468L) + x1 + (156L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (624L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (624L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_70 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(208L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_71 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)), static_cast<long>(624L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (624L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (624L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_72 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(312L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)), static_cast<long>(312L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(61152L + x2 + (196L*x1) + (196L*x1_inner) + (122304L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (312L*x2) + (61152L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(104L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (104L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(52);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (52L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(104);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-52L) + x1 + (52L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (104L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_silu_74 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(978432L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                tmp2.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_silu_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(624L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (624L*x2) + (122304L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (624L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(4992L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(196.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_silu_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(416L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_77 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(624L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (624L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (624L*x1) + (122304L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_78 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_79 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(240);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(480);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-240L) + x1 + (240L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_80 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_81 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_82 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_83 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(120);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(240);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-120L) + x1 + (120L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(360);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-240L) + x1 + (120L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(480);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-360L) + x1 + (120L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_85 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (480L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_86 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_87 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(80);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_88 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(240);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(480);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-240L) + x1 + (240L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_89 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_90 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_91 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_92 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(120);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(240);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-120L) + x1 + (120L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(360);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-240L) + x1 + (120L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(480);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-360L) + x1 + (120L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_94 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (480L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_95 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(80);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_97 = async_compile.cpp('''
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
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(240);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(480);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-240L) + x1 + (240L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp28;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr1[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr2 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_98 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(23520L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_99 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_100 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)), static_cast<long>(120L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(70560L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (120L*x2) + (23520L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_101 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(120);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (120L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(240);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-120L) + x1 + (120L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(360);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-240L) + x1 + (120L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(480);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-360L) + x1 + (120L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (480L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_102 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_103 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)), static_cast<long>(480L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (480L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_104 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (94080L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(1L))
                {
                    auto tmp15 = in_ptr2[static_cast<long>(x1)];
                    auto tmp17 = in_ptr3[static_cast<long>(x1)];
                    auto tmp25 = in_ptr4[static_cast<long>(x1)];
                    auto tmp27 = in_ptr5[static_cast<long>(x1)];
                    auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (160L*x0))];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(80);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (80L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(160);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = [&]
                    {
                        auto tmp12 = in_ptr1[static_cast<long>((-80L) + x1 + (80L*x0))];
                        return tmp12;
                    }
                    ;
                    auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                    auto tmp14 = tmp4 ? tmp7 : tmp13;
                    auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                    auto tmp18 = static_cast<float>(1e-05);
                    auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                    auto tmp20 = std::sqrt(tmp19);
                    auto tmp21 = 1 / tmp20;
                    auto tmp22 = static_cast<float>(1.0);
                    auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                    auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                    auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                    auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    in_out_ptr0[static_cast<long>(x1 + (160L*x0))] = tmp30;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (188160L*x0)), static_cast<long>(960L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (188160L*x0)), static_cast<long>(960L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x2) + (188160L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_107 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(47040L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_108 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(94080L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(94080L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_109 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(192L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(141120L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)), static_cast<long>(240L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(192L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(141120L + x2 + (196L*x1) + (196L*x1_inner) + (188160L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (240L*x2) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_110 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(240);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (240L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(480);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-240L) + x1 + (240L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(720);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-480L) + x1 + (240L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(960);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-720L) + x1 + (240L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (960L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(7680L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_mul_sigmoid_silu_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        tmp5.store(in_out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_113 = async_compile.cpp('''
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (264L*x0)));
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_114 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))];
                        out_ptr1[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_115 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_116 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_117 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_118 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(396);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (396L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(792);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-396L) + x1 + (396L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(1188);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-792L) + x1 + (396L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(1584);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-1188L) + x1 + (396L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_120 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (1584L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_121 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_122 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(1L))
            {
                auto tmp15 = in_ptr2[static_cast<long>(x1)];
                auto tmp17 = in_ptr3[static_cast<long>(x1)];
                auto tmp25 = in_ptr4[static_cast<long>(x1)];
                auto tmp27 = in_ptr5[static_cast<long>(x1)];
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (264L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(132);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (132L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(264);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-132L) + x1 + (132L*x0))];
                    return tmp12;
                }
                ;
                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp14 = tmp4 ? tmp7 : tmp13;
                auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                auto tmp18 = static_cast<float>(1e-05);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp20 = std::sqrt(tmp19);
                auto tmp21 = 1 / tmp20;
                auto tmp22 = static_cast<float>(1.0);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (264L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_123 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))];
                        out_ptr1[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_124 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_125 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_126 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_127 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(396);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (396L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(792);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-396L) + x1 + (396L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(1188);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-792L) + x1 + (396L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(1584);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-1188L) + x1 + (396L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_128 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_129 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (1584L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_130 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_131 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(1L))
            {
                auto tmp15 = in_ptr2[static_cast<long>(x1)];
                auto tmp17 = in_ptr3[static_cast<long>(x1)];
                auto tmp25 = in_ptr4[static_cast<long>(x1)];
                auto tmp27 = in_ptr5[static_cast<long>(x1)];
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (264L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(132);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (132L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(264);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-132L) + x1 + (132L*x0))];
                    return tmp12;
                }
                ;
                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp14 = tmp4 ? tmp7 : tmp13;
                auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                auto tmp18 = static_cast<float>(1e-05);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp20 = std::sqrt(tmp19);
                auto tmp21 = 1 / tmp20;
                auto tmp22 = static_cast<float>(1.0);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (264L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_132 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (1584L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            tmp3.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp2.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = out_ptr0[static_cast<long>(x2 + (49L*x1) + (77616L*x0))];
                        out_ptr1[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_133 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(19404L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_134 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_135 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(392L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)), static_cast<long>(396L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (396L*x2) + (19404L*x0)));
                    }
                }
                #pragma GCC ivdep
                for(long x1=static_cast<long>(392L); x1<static_cast<long>(396L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(58212L + x2 + (49L*x1) + (77616L*x0))];
                        out_ptr0[static_cast<long>(x1 + (396L*x2) + (19404L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_136 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(1L))
                {
                    auto tmp31 = in_ptr4[static_cast<long>(x1)];
                    auto tmp33 = in_ptr5[static_cast<long>(x1)];
                    auto tmp41 = in_ptr6[static_cast<long>(x1)];
                    auto tmp43 = in_ptr7[static_cast<long>(x1)];
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = static_cast<long>(0);
                    auto tmp2 = tmp0 >= tmp1;
                    auto tmp3 = static_cast<long>(396);
                    auto tmp4 = tmp0 < tmp3;
                    auto tmp5 = [&]
                    {
                        auto tmp6 = in_ptr0[static_cast<long>(x1 + (396L*x0))];
                        return tmp6;
                    }
                    ;
                    auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                    auto tmp8 = tmp0 >= tmp3;
                    auto tmp9 = static_cast<long>(792);
                    auto tmp10 = tmp0 < tmp9;
                    auto tmp11 = tmp8 & tmp10;
                    auto tmp12 = [&]
                    {
                        auto tmp13 = in_ptr1[static_cast<long>((-396L) + x1 + (396L*x0))];
                        return tmp13;
                    }
                    ;
                    auto tmp14 = tmp11 ? tmp12() : static_cast<decltype(tmp12())>(0.0);
                    auto tmp15 = tmp0 >= tmp9;
                    auto tmp16 = static_cast<long>(1188);
                    auto tmp17 = tmp0 < tmp16;
                    auto tmp18 = tmp15 & tmp17;
                    auto tmp19 = [&]
                    {
                        auto tmp20 = in_ptr2[static_cast<long>((-792L) + x1 + (396L*x0))];
                        return tmp20;
                    }
                    ;
                    auto tmp21 = tmp18 ? tmp19() : static_cast<decltype(tmp19())>(0.0);
                    auto tmp22 = tmp0 >= tmp16;
                    auto tmp23 = static_cast<long>(1584);
                    auto tmp24 = tmp0 < tmp23;
                    auto tmp25 = [&]
                    {
                        auto tmp26 = in_ptr3[static_cast<long>((-1188L) + x1 + (396L*x0))];
                        return tmp26;
                    }
                    ;
                    auto tmp27 = tmp22 ? tmp25() : static_cast<decltype(tmp25())>(0.0);
                    auto tmp28 = tmp18 ? tmp21 : tmp27;
                    auto tmp29 = tmp11 ? tmp14 : tmp28;
                    auto tmp30 = tmp4 ? tmp7 : tmp29;
                    auto tmp32 = decltype(tmp30)(tmp30 - tmp31);
                    auto tmp34 = static_cast<float>(1e-05);
                    auto tmp35 = decltype(tmp33)(tmp33 + tmp34);
                    auto tmp36 = std::sqrt(tmp35);
                    auto tmp37 = 1 / tmp36;
                    auto tmp38 = static_cast<float>(1.0);
                    auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                    auto tmp40 = decltype(tmp32)(tmp32 * tmp39);
                    auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                    auto tmp44 = decltype(tmp42)(tmp42 + tmp43);
                    out_ptr0[static_cast<long>(x1 + (1584L*x0))] = tmp44;
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12672L); x0+=static_cast<long>(8L))
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


cpp_fused_silu_137 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1056L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_mul_sigmoid_silu_138 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1584L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp0[8*8] __attribute__ ((aligned (8)));
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        at::vec::transpose_mxn<float,8,8>(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)), static_cast<long>(1584L), tmp0, 8);
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp1 = at::vec::Vectorized<float>::loadu(tmp0 + static_cast<long>(8L*x1_inner));
                            auto tmp4 = in_ptr1[static_cast<long>(x1 + x1_inner + (1584L*x0))];
                            auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                            auto tmp3 = tmp1 * tmp2;
                            auto tmp5 = decltype(tmp4)(1) / (decltype(tmp4)(1) + std::exp(-tmp4));
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = tmp3 * tmp6;
                            tmp7.store(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                        }
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1584L*x2) + (77616L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (1584L*x0)));
                        auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                        auto tmp2 = tmp0 * tmp1;
                        auto tmp4 = decltype(tmp3)(1)/(decltype(tmp3)(1) + tmp3.neg().exp());
                        auto tmp5 = tmp2 * tmp4;
                        { __at_align__ float tmpbuf[8*sizeof(float)/sizeof(float)]; tmp5.store(tmpbuf); for (long x1_inner = 0; x1_inner < 8; x1_inner++) out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))] = tmpbuf[x1_inner]; }
                    }
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = out_ptr0[static_cast<long>(x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr1 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_139 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(792L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(48L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)), static_cast<long>(792L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(48L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(38808L + x2 + (49L*x1) + (49L*x1_inner) + (77616L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (792L*x2) + (38808L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_140 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(264L); x1+=static_cast<long>(1L))
            {
                auto tmp15 = in_ptr2[static_cast<long>(x1)];
                auto tmp17 = in_ptr3[static_cast<long>(x1)];
                auto tmp25 = in_ptr4[static_cast<long>(x1)];
                auto tmp27 = in_ptr5[static_cast<long>(x1)];
                auto tmp29 = in_out_ptr0[static_cast<long>(x1 + (264L*x0))];
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(132);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = in_ptr0[static_cast<long>(x1 + (132L*x0))];
                    return tmp6;
                }
                ;
                auto tmp7 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp8 = tmp0 >= tmp3;
                auto tmp9 = static_cast<long>(264);
                auto tmp10 = tmp0 < tmp9;
                auto tmp11 = [&]
                {
                    auto tmp12 = in_ptr1[static_cast<long>((-132L) + x1 + (132L*x0))];
                    return tmp12;
                }
                ;
                auto tmp13 = tmp8 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                auto tmp14 = tmp4 ? tmp7 : tmp13;
                auto tmp16 = decltype(tmp14)(tmp14 - tmp15);
                auto tmp18 = static_cast<float>(1e-05);
                auto tmp19 = decltype(tmp17)(tmp17 + tmp18);
                auto tmp20 = std::sqrt(tmp19);
                auto tmp21 = 1 / tmp20;
                auto tmp22 = static_cast<float>(1.0);
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = decltype(tmp16)(tmp16 * tmp23);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp28 = decltype(tmp26)(tmp26 + tmp27);
                auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                in_out_ptr0[static_cast<long>(x1 + (264L*x0))] = tmp30;
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_mean_relu_141 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1536L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1536L*x2) + (75264L*x0)));
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
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1536L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(12288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (192, ), (1, ))
    assert_size_stride(arg7_1, (192, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (40, ), (1, ))
    assert_size_stride(arg11_1, (40, ), (1, ))
    assert_size_stride(arg12_1, (120, ), (1, ))
    assert_size_stride(arg13_1, (120, ), (1, ))
    assert_size_stride(arg14_1, (120, ), (1, ))
    assert_size_stride(arg15_1, (120, ), (1, ))
    assert_size_stride(arg16_1, (40, ), (1, ))
    assert_size_stride(arg17_1, (40, ), (1, ))
    assert_size_stride(arg18_1, (240, ), (1, ))
    assert_size_stride(arg19_1, (240, ), (1, ))
    assert_size_stride(arg20_1, (240, ), (1, ))
    assert_size_stride(arg21_1, (240, ), (1, ))
    assert_size_stride(arg22_1, (56, ), (1, ))
    assert_size_stride(arg23_1, (56, ), (1, ))
    assert_size_stride(arg24_1, (336, ), (1, ))
    assert_size_stride(arg25_1, (336, ), (1, ))
    assert_size_stride(arg26_1, (336, ), (1, ))
    assert_size_stride(arg27_1, (336, ), (1, ))
    assert_size_stride(arg28_1, (56, ), (1, ))
    assert_size_stride(arg29_1, (56, ), (1, ))
    assert_size_stride(arg30_1, (336, ), (1, ))
    assert_size_stride(arg31_1, (336, ), (1, ))
    assert_size_stride(arg32_1, (336, ), (1, ))
    assert_size_stride(arg33_1, (336, ), (1, ))
    assert_size_stride(arg34_1, (56, ), (1, ))
    assert_size_stride(arg35_1, (56, ), (1, ))
    assert_size_stride(arg36_1, (336, ), (1, ))
    assert_size_stride(arg37_1, (336, ), (1, ))
    assert_size_stride(arg38_1, (336, ), (1, ))
    assert_size_stride(arg39_1, (336, ), (1, ))
    assert_size_stride(arg40_1, (56, ), (1, ))
    assert_size_stride(arg41_1, (56, ), (1, ))
    assert_size_stride(arg42_1, (336, ), (1, ))
    assert_size_stride(arg43_1, (336, ), (1, ))
    assert_size_stride(arg44_1, (336, ), (1, ))
    assert_size_stride(arg45_1, (336, ), (1, ))
    assert_size_stride(arg46_1, (104, ), (1, ))
    assert_size_stride(arg47_1, (104, ), (1, ))
    assert_size_stride(arg48_1, (624, ), (1, ))
    assert_size_stride(arg49_1, (624, ), (1, ))
    assert_size_stride(arg50_1, (624, ), (1, ))
    assert_size_stride(arg51_1, (624, ), (1, ))
    assert_size_stride(arg52_1, (104, ), (1, ))
    assert_size_stride(arg53_1, (104, ), (1, ))
    assert_size_stride(arg54_1, (624, ), (1, ))
    assert_size_stride(arg55_1, (624, ), (1, ))
    assert_size_stride(arg56_1, (624, ), (1, ))
    assert_size_stride(arg57_1, (624, ), (1, ))
    assert_size_stride(arg58_1, (104, ), (1, ))
    assert_size_stride(arg59_1, (104, ), (1, ))
    assert_size_stride(arg60_1, (624, ), (1, ))
    assert_size_stride(arg61_1, (624, ), (1, ))
    assert_size_stride(arg62_1, (624, ), (1, ))
    assert_size_stride(arg63_1, (624, ), (1, ))
    assert_size_stride(arg64_1, (104, ), (1, ))
    assert_size_stride(arg65_1, (104, ), (1, ))
    assert_size_stride(arg66_1, (624, ), (1, ))
    assert_size_stride(arg67_1, (624, ), (1, ))
    assert_size_stride(arg68_1, (624, ), (1, ))
    assert_size_stride(arg69_1, (624, ), (1, ))
    assert_size_stride(arg70_1, (160, ), (1, ))
    assert_size_stride(arg71_1, (160, ), (1, ))
    assert_size_stride(arg72_1, (480, ), (1, ))
    assert_size_stride(arg73_1, (480, ), (1, ))
    assert_size_stride(arg74_1, (480, ), (1, ))
    assert_size_stride(arg75_1, (480, ), (1, ))
    assert_size_stride(arg76_1, (160, ), (1, ))
    assert_size_stride(arg77_1, (160, ), (1, ))
    assert_size_stride(arg78_1, (480, ), (1, ))
    assert_size_stride(arg79_1, (480, ), (1, ))
    assert_size_stride(arg80_1, (480, ), (1, ))
    assert_size_stride(arg81_1, (480, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (480, ), (1, ))
    assert_size_stride(arg85_1, (480, ), (1, ))
    assert_size_stride(arg86_1, (480, ), (1, ))
    assert_size_stride(arg87_1, (480, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (960, ), (1, ))
    assert_size_stride(arg91_1, (960, ), (1, ))
    assert_size_stride(arg92_1, (960, ), (1, ))
    assert_size_stride(arg93_1, (960, ), (1, ))
    assert_size_stride(arg94_1, (264, ), (1, ))
    assert_size_stride(arg95_1, (264, ), (1, ))
    assert_size_stride(arg96_1, (1584, ), (1, ))
    assert_size_stride(arg97_1, (1584, ), (1, ))
    assert_size_stride(arg98_1, (1584, ), (1, ))
    assert_size_stride(arg99_1, (1584, ), (1, ))
    assert_size_stride(arg100_1, (264, ), (1, ))
    assert_size_stride(arg101_1, (264, ), (1, ))
    assert_size_stride(arg102_1, (1584, ), (1, ))
    assert_size_stride(arg103_1, (1584, ), (1, ))
    assert_size_stride(arg104_1, (1584, ), (1, ))
    assert_size_stride(arg105_1, (1584, ), (1, ))
    assert_size_stride(arg106_1, (264, ), (1, ))
    assert_size_stride(arg107_1, (264, ), (1, ))
    assert_size_stride(arg108_1, (1584, ), (1, ))
    assert_size_stride(arg109_1, (1584, ), (1, ))
    assert_size_stride(arg110_1, (1584, ), (1, ))
    assert_size_stride(arg111_1, (1584, ), (1, ))
    assert_size_stride(arg112_1, (264, ), (1, ))
    assert_size_stride(arg113_1, (264, ), (1, ))
    assert_size_stride(arg114_1, (1536, ), (1, ))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg117_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg119_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg120_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg121_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg122_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg123_1, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg124_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg125_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg126_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg127_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg128_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg129_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg130_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg131_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg132_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg134_1, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg135_1, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg136_1, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg137_1, (20, ), (1, ))
    assert_size_stride(arg138_1, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg139_1, (240, ), (1, ))
    assert_size_stride(arg140_1, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg141_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg142_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg143_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg144_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg146_1, (28, ), (1, ))
    assert_size_stride(arg147_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg148_1, (336, ), (1, ))
    assert_size_stride(arg149_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg150_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg151_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg152_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg153_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg155_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg156_1, (28, ), (1, ))
    assert_size_stride(arg157_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg158_1, (336, ), (1, ))
    assert_size_stride(arg159_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg160_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg161_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg162_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg163_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg164_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg165_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg166_1, (28, ), (1, ))
    assert_size_stride(arg167_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg168_1, (336, ), (1, ))
    assert_size_stride(arg169_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg170_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg171_1, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg172_1, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg173_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg174_1, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg175_1, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg176_1, (14, ), (1, ))
    assert_size_stride(arg177_1, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg178_1, (336, ), (1, ))
    assert_size_stride(arg179_1, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg180_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg181_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg182_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg183_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg184_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg185_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg186_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg187_1, (26, ), (1, ))
    assert_size_stride(arg188_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg189_1, (624, ), (1, ))
    assert_size_stride(arg190_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg191_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg192_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg193_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg194_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg195_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg196_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg197_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg198_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg199_1, (26, ), (1, ))
    assert_size_stride(arg200_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg201_1, (624, ), (1, ))
    assert_size_stride(arg202_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg203_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg204_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg205_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg206_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg207_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg208_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg209_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg210_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg211_1, (26, ), (1, ))
    assert_size_stride(arg212_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg213_1, (624, ), (1, ))
    assert_size_stride(arg214_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg215_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg216_1, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg217_1, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg218_1, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg219_1, (52, ), (1, ))
    assert_size_stride(arg220_1, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg221_1, (624, ), (1, ))
    assert_size_stride(arg222_1, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg223_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg224_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg225_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg226_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg227_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg228_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg229_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg230_1, (80, ), (1, ))
    assert_size_stride(arg231_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg232_1, (480, ), (1, ))
    assert_size_stride(arg233_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg234_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg235_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg236_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg237_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg238_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg239_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg240_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg241_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg242_1, (80, ), (1, ))
    assert_size_stride(arg243_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg244_1, (480, ), (1, ))
    assert_size_stride(arg245_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg246_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg247_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg248_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg249_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg250_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg251_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg252_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg253_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg254_1, (80, ), (1, ))
    assert_size_stride(arg255_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg256_1, (480, ), (1, ))
    assert_size_stride(arg257_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg258_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg259_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg260_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg261_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg262_1, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg263_1, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg264_1, (80, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg265_1, (80, ), (1, ))
    assert_size_stride(arg266_1, (960, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg267_1, (960, ), (1, ))
    assert_size_stride(arg268_1, (264, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg269_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg270_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg271_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg272_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg273_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg274_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg275_1, (132, ), (1, ))
    assert_size_stride(arg276_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg277_1, (1584, ), (1, ))
    assert_size_stride(arg278_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg279_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg280_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg281_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg282_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg283_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg284_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg285_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg286_1, (132, ), (1, ))
    assert_size_stride(arg287_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg288_1, (1584, ), (1, ))
    assert_size_stride(arg289_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg290_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg291_1, (1584, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg292_1, (396, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg293_1, (396, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg294_1, (396, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg295_1, (396, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg296_1, (132, 1584, 1, 1), (1584, 1, 1, 1))
    assert_size_stride(arg297_1, (132, ), (1, ))
    assert_size_stride(arg298_1, (1584, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(arg299_1, (1584, ), (1, ))
    assert_size_stride(arg300_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg301_1, (132, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(arg302_1, (1536, 264, 1, 1), (264, 1, 1, 1))
    assert_size_stride(arg303_1, (1000, 1536), (1536, 1))
    assert_size_stride(arg304_1, (1000, ), (1, ))
    assert_size_stride(arg305_1, (32, ), (1, ))
    assert_size_stride(arg306_1, (32, ), (1, ))
    assert_size_stride(arg307_1, (32, ), (1, ))
    assert_size_stride(arg308_1, (32, ), (1, ))
    assert_size_stride(arg309_1, (32, ), (1, ))
    assert_size_stride(arg310_1, (32, ), (1, ))
    assert_size_stride(arg311_1, (192, ), (1, ))
    assert_size_stride(arg312_1, (192, ), (1, ))
    assert_size_stride(arg313_1, (192, ), (1, ))
    assert_size_stride(arg314_1, (192, ), (1, ))
    assert_size_stride(arg315_1, (40, ), (1, ))
    assert_size_stride(arg316_1, (40, ), (1, ))
    assert_size_stride(arg317_1, (120, ), (1, ))
    assert_size_stride(arg318_1, (120, ), (1, ))
    assert_size_stride(arg319_1, (120, ), (1, ))
    assert_size_stride(arg320_1, (120, ), (1, ))
    assert_size_stride(arg321_1, (40, ), (1, ))
    assert_size_stride(arg322_1, (40, ), (1, ))
    assert_size_stride(arg323_1, (240, ), (1, ))
    assert_size_stride(arg324_1, (240, ), (1, ))
    assert_size_stride(arg325_1, (240, ), (1, ))
    assert_size_stride(arg326_1, (240, ), (1, ))
    assert_size_stride(arg327_1, (56, ), (1, ))
    assert_size_stride(arg328_1, (56, ), (1, ))
    assert_size_stride(arg329_1, (336, ), (1, ))
    assert_size_stride(arg330_1, (336, ), (1, ))
    assert_size_stride(arg331_1, (336, ), (1, ))
    assert_size_stride(arg332_1, (336, ), (1, ))
    assert_size_stride(arg333_1, (56, ), (1, ))
    assert_size_stride(arg334_1, (56, ), (1, ))
    assert_size_stride(arg335_1, (336, ), (1, ))
    assert_size_stride(arg336_1, (336, ), (1, ))
    assert_size_stride(arg337_1, (336, ), (1, ))
    assert_size_stride(arg338_1, (336, ), (1, ))
    assert_size_stride(arg339_1, (56, ), (1, ))
    assert_size_stride(arg340_1, (56, ), (1, ))
    assert_size_stride(arg341_1, (336, ), (1, ))
    assert_size_stride(arg342_1, (336, ), (1, ))
    assert_size_stride(arg343_1, (336, ), (1, ))
    assert_size_stride(arg344_1, (336, ), (1, ))
    assert_size_stride(arg345_1, (56, ), (1, ))
    assert_size_stride(arg346_1, (56, ), (1, ))
    assert_size_stride(arg347_1, (336, ), (1, ))
    assert_size_stride(arg348_1, (336, ), (1, ))
    assert_size_stride(arg349_1, (336, ), (1, ))
    assert_size_stride(arg350_1, (336, ), (1, ))
    assert_size_stride(arg351_1, (104, ), (1, ))
    assert_size_stride(arg352_1, (104, ), (1, ))
    assert_size_stride(arg353_1, (624, ), (1, ))
    assert_size_stride(arg354_1, (624, ), (1, ))
    assert_size_stride(arg355_1, (624, ), (1, ))
    assert_size_stride(arg356_1, (624, ), (1, ))
    assert_size_stride(arg357_1, (104, ), (1, ))
    assert_size_stride(arg358_1, (104, ), (1, ))
    assert_size_stride(arg359_1, (624, ), (1, ))
    assert_size_stride(arg360_1, (624, ), (1, ))
    assert_size_stride(arg361_1, (624, ), (1, ))
    assert_size_stride(arg362_1, (624, ), (1, ))
    assert_size_stride(arg363_1, (104, ), (1, ))
    assert_size_stride(arg364_1, (104, ), (1, ))
    assert_size_stride(arg365_1, (624, ), (1, ))
    assert_size_stride(arg366_1, (624, ), (1, ))
    assert_size_stride(arg367_1, (624, ), (1, ))
    assert_size_stride(arg368_1, (624, ), (1, ))
    assert_size_stride(arg369_1, (104, ), (1, ))
    assert_size_stride(arg370_1, (104, ), (1, ))
    assert_size_stride(arg371_1, (624, ), (1, ))
    assert_size_stride(arg372_1, (624, ), (1, ))
    assert_size_stride(arg373_1, (624, ), (1, ))
    assert_size_stride(arg374_1, (624, ), (1, ))
    assert_size_stride(arg375_1, (160, ), (1, ))
    assert_size_stride(arg376_1, (160, ), (1, ))
    assert_size_stride(arg377_1, (480, ), (1, ))
    assert_size_stride(arg378_1, (480, ), (1, ))
    assert_size_stride(arg379_1, (480, ), (1, ))
    assert_size_stride(arg380_1, (480, ), (1, ))
    assert_size_stride(arg381_1, (160, ), (1, ))
    assert_size_stride(arg382_1, (160, ), (1, ))
    assert_size_stride(arg383_1, (480, ), (1, ))
    assert_size_stride(arg384_1, (480, ), (1, ))
    assert_size_stride(arg385_1, (480, ), (1, ))
    assert_size_stride(arg386_1, (480, ), (1, ))
    assert_size_stride(arg387_1, (160, ), (1, ))
    assert_size_stride(arg388_1, (160, ), (1, ))
    assert_size_stride(arg389_1, (480, ), (1, ))
    assert_size_stride(arg390_1, (480, ), (1, ))
    assert_size_stride(arg391_1, (480, ), (1, ))
    assert_size_stride(arg392_1, (480, ), (1, ))
    assert_size_stride(arg393_1, (160, ), (1, ))
    assert_size_stride(arg394_1, (160, ), (1, ))
    assert_size_stride(arg395_1, (960, ), (1, ))
    assert_size_stride(arg396_1, (960, ), (1, ))
    assert_size_stride(arg397_1, (960, ), (1, ))
    assert_size_stride(arg398_1, (960, ), (1, ))
    assert_size_stride(arg399_1, (264, ), (1, ))
    assert_size_stride(arg400_1, (264, ), (1, ))
    assert_size_stride(arg401_1, (1584, ), (1, ))
    assert_size_stride(arg402_1, (1584, ), (1, ))
    assert_size_stride(arg403_1, (1584, ), (1, ))
    assert_size_stride(arg404_1, (1584, ), (1, ))
    assert_size_stride(arg405_1, (264, ), (1, ))
    assert_size_stride(arg406_1, (264, ), (1, ))
    assert_size_stride(arg407_1, (1584, ), (1, ))
    assert_size_stride(arg408_1, (1584, ), (1, ))
    assert_size_stride(arg409_1, (1584, ), (1, ))
    assert_size_stride(arg410_1, (1584, ), (1, ))
    assert_size_stride(arg411_1, (264, ), (1, ))
    assert_size_stride(arg412_1, (264, ), (1, ))
    assert_size_stride(arg413_1, (1584, ), (1, ))
    assert_size_stride(arg414_1, (1584, ), (1, ))
    assert_size_stride(arg415_1, (1584, ), (1, ))
    assert_size_stride(arg416_1, (1584, ), (1, ))
    assert_size_stride(arg417_1, (264, ), (1, ))
    assert_size_stride(arg418_1, (264, ), (1, ))
    assert_size_stride(arg419_1, (1536, ), (1, ))
    assert_size_stride(arg420_1, (1536, ), (1, ))
    assert_size_stride(arg421_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg421_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg116_1
    del arg421_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()))
    del arg0_1
    del arg1_1
    del arg305_1
    del arg306_1
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf4, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del arg117_1
    buf5 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg2_1
    del arg307_1
    del arg308_1
    del arg3_1
    # Source Nodes: [x_11, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del arg118_1
    buf7 = reinterpret_tensor(buf5, (8, 32, 112, 112), (401408, 12544, 112, 1), 0); del buf5  # reuse
    buf8 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_3(c_void_p(buf6.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg309_1
    del arg310_1
    del arg4_1
    del arg5_1
    del buf3
    del buf6
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf8, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg119_1
    buf10 = buf8; del buf8  # reuse
    cpp_fused_convolution_4(c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf7
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg120_1
    del buf10
    buf12 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_5(c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(buf12.data_ptr()))
    del arg311_1
    del arg312_1
    del arg6_1
    del arg7_1
    del buf11
    del buf9
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_0], Original ATen: [aten.convolution]
    buf13 = extern_kernels.convolution(reinterpret_tensor(buf12, (8, 64, 112, 112), (2408448, 1, 21504, 192), 0), arg121_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf13, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg121_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_1], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(reinterpret_tensor(buf12, (8, 64, 112, 112), (2408448, 1, 21504, 192), 64), arg122_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf14, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg122_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_dw_2], Original ATen: [aten.convolution]
    buf15 = extern_kernels.convolution(reinterpret_tensor(buf12, (8, 64, 112, 112), (2408448, 1, 21504, 192), 128), arg123_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf15, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg123_1
    del buf12
    buf16 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_6(c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf15.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()), c_void_p(buf16.data_ptr()))
    del arg313_1
    del arg314_1
    del arg8_1
    del arg9_1
    del buf13
    del buf14
    del buf15
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(reinterpret_tensor(buf16, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf17, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg124_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
    buf18 = extern_kernels.convolution(reinterpret_tensor(buf16, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf18, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg125_1
    del buf16
    buf19 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_7(c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg10_1
    del arg11_1
    del arg315_1
    del arg316_1
    del buf17
    del buf18
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 60, 56, 56), (188160, 1, 3360, 60))
    del arg126_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 60, 56, 56), (188160, 1, 3360, 60))
    del arg127_1
    buf22 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_8(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg12_1
    del arg13_1
    del arg317_1
    del arg318_1
    del buf20
    # Source Nodes: [cat_78, x_38, x_41, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf23 = extern_kernels.convolution(buf22, arg128_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf23, (8, 120, 56, 56), (376320, 1, 6720, 120))
    del arg128_1
    del buf22
    buf24 = buf23; del buf23  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_9(c_void_p(buf24.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg319_1
    del arg320_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf25 = extern_kernels.convolution(reinterpret_tensor(buf24, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf25, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg129_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf26 = extern_kernels.convolution(reinterpret_tensor(buf24, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf26, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg130_1
    del buf24
    buf27 = buf19; del buf19  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_10(c_void_p(buf27.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg16_1
    del arg17_1
    del arg321_1
    del arg322_1
    del buf25
    del buf26
    # Source Nodes: [cat_77, shortcut_3, x_50, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf28 = extern_kernels.convolution(buf27, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 240, 56, 56), (752640, 1, 13440, 240))
    del arg131_1
    del buf27
    buf29 = buf28; del buf28  # reuse
    buf30 = empty((8, 240, 56, 56), device='cpu', dtype=torch.float32)
    buf31 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_11(c_void_p(buf29.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()))
    del arg18_1
    del arg19_1
    del arg323_1
    del arg324_1
    del buf29
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_0], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg132_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf32, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg132_1
    buf33 = buf31; del buf31  # reuse
    cpp_fused_convolution_12(c_void_p(buf30.data_ptr()), c_void_p(buf33.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_1], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf33, arg133_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf34, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg133_1
    buf35 = buf33; del buf33  # reuse
    cpp_fused_convolution_13(c_void_p(buf30.data_ptr()), c_void_p(buf35.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_2], Original ATen: [aten.convolution]
    buf36 = extern_kernels.convolution(buf35, arg134_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf36, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg134_1
    buf37 = buf35; del buf35  # reuse
    cpp_fused_convolution_14(c_void_p(buf30.data_ptr()), c_void_p(buf37.data_ptr()))
    del buf30
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___conv_dw_3], Original ATen: [aten.convolution]
    buf38 = extern_kernels.convolution(buf37, arg135_1, stride=(2, 2), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf38, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg135_1
    buf39 = reinterpret_tensor(buf37, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf37  # reuse
    buf40 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf41 = reinterpret_tensor(buf40, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_15(c_void_p(buf41.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf39.data_ptr()))
    del arg20_1
    del arg21_1
    del arg325_1
    del arg326_1
    del buf32
    del buf34
    del buf36
    del buf38
    # Source Nodes: [x_65, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf42 = extern_kernels.convolution(buf41, arg136_1, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf42, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg136_1
    del arg137_1
    del buf41
    buf43 = buf42; del buf42  # reuse
    cpp_fused_silu_16(c_void_p(buf43.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.silu]
    buf44 = extern_kernels.convolution(buf43, arg138_1, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg138_1
    del arg139_1
    del buf43
    buf45 = buf39; del buf39  # reuse
    cpp_fused_mul_sigmoid_silu_17(c_void_p(buf45.data_ptr()), c_void_p(buf44.data_ptr()))
    del buf44
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_65, x_66, x_67], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf46 = extern_kernels.convolution(buf45, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf46, (8, 56, 28, 28), (43904, 1, 1568, 56))
    del arg140_1
    buf47 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_18(c_void_p(buf47.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()))
    del arg22_1
    del arg23_1
    del arg327_1
    del arg328_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf48 = extern_kernels.convolution(reinterpret_tensor(buf47, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg141_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(reinterpret_tensor(buf47, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg142_1
    buf50 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    buf51 = empty((8, 336, 28, 28), device='cpu', dtype=torch.float32)
    buf52 = empty_strided((8, 168, 28, 28), (131712, 1, 4704, 168), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_19(c_void_p(buf48.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()), c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()))
    del arg24_1
    del arg25_1
    del arg329_1
    del arg330_1
    del buf48
    del buf49
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf53 = extern_kernels.convolution(buf52, arg143_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf53, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg143_1
    buf54 = buf52; del buf52  # reuse
    cpp_fused_convolution_20(c_void_p(buf51.data_ptr()), c_void_p(buf54.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, arg144_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf55, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg144_1
    del buf54
    buf56 = reinterpret_tensor(buf51, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf51  # reuse
    buf57 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf58 = reinterpret_tensor(buf57, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf57  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_21(c_void_p(buf58.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf56.data_ptr()))
    del arg26_1
    del arg27_1
    del arg331_1
    del arg332_1
    del buf53
    # Source Nodes: [x_83, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf59 = extern_kernels.convolution(buf58, arg145_1, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf59, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg145_1
    del arg146_1
    del buf58
    buf60 = buf59; del buf59  # reuse
    cpp_fused_silu_22(c_void_p(buf60.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.silu]
    buf61 = extern_kernels.convolution(buf60, arg147_1, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf61, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg147_1
    del arg148_1
    del buf60
    buf62 = reinterpret_tensor(buf50, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf50  # reuse
    buf63 = buf55; del buf55  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_23(c_void_p(buf56.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf63, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg149_1
    buf65 = buf63; del buf63  # reuse
    cpp_fused_convolution_24(c_void_p(buf62.data_ptr()), c_void_p(buf65.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg150_1
    buf67 = buf47; del buf47  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_25(c_void_p(buf67.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()))
    del arg28_1
    del arg29_1
    del arg333_1
    del arg334_1
    del buf64
    del buf66
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(reinterpret_tensor(buf67, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg151_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(reinterpret_tensor(buf67, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf69, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg152_1
    buf70 = reinterpret_tensor(buf62, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf62  # reuse
    buf71 = reinterpret_tensor(buf56, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf56  # reuse
    buf72 = buf65; del buf65  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_26(c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()), c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(buf72.data_ptr()))
    del arg30_1
    del arg31_1
    del arg335_1
    del arg336_1
    del buf68
    del buf69
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf73 = extern_kernels.convolution(buf72, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf73, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg153_1
    buf74 = buf72; del buf72  # reuse
    cpp_fused_convolution_27(c_void_p(buf71.data_ptr()), c_void_p(buf74.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, arg154_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf75, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg154_1
    del buf74
    buf76 = reinterpret_tensor(buf71, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf71  # reuse
    buf77 = reinterpret_tensor(buf61, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf61  # reuse
    buf78 = reinterpret_tensor(buf77, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf77  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_28(c_void_p(buf78.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf76.data_ptr()))
    del arg32_1
    del arg337_1
    del arg338_1
    del arg33_1
    del buf73
    # Source Nodes: [x_103, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf79 = extern_kernels.convolution(buf78, arg155_1, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf79, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg155_1
    del arg156_1
    del buf78
    buf80 = buf79; del buf79  # reuse
    cpp_fused_silu_29(c_void_p(buf80.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.silu]
    buf81 = extern_kernels.convolution(buf80, arg157_1, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf81, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg157_1
    del arg158_1
    del buf80
    buf82 = reinterpret_tensor(buf70, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf70  # reuse
    buf83 = buf75; del buf75  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_30(c_void_p(buf76.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf83, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg159_1
    buf85 = buf83; del buf83  # reuse
    cpp_fused_convolution_31(c_void_p(buf82.data_ptr()), c_void_p(buf85.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg160_1
    buf87 = buf67; del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_32(c_void_p(buf87.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()))
    del arg339_1
    del arg340_1
    del arg34_1
    del arg35_1
    del buf84
    del buf86
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(reinterpret_tensor(buf87, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg161_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(reinterpret_tensor(buf87, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg162_1
    buf90 = reinterpret_tensor(buf82, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf82  # reuse
    buf91 = reinterpret_tensor(buf76, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf76  # reuse
    buf92 = buf85; del buf85  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_33(c_void_p(buf88.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()))
    del arg341_1
    del arg342_1
    del arg36_1
    del arg37_1
    del buf88
    del buf89
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf93 = extern_kernels.convolution(buf92, arg163_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf93, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg163_1
    buf94 = buf92; del buf92  # reuse
    cpp_fused_convolution_34(c_void_p(buf91.data_ptr()), c_void_p(buf94.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg164_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf95, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg164_1
    del buf94
    buf96 = reinterpret_tensor(buf91, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf91  # reuse
    buf97 = reinterpret_tensor(buf81, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf81  # reuse
    buf98 = reinterpret_tensor(buf97, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf97  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_35(c_void_p(buf98.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf96.data_ptr()))
    del arg343_1
    del arg344_1
    del arg38_1
    del arg39_1
    del buf93
    # Source Nodes: [x_123, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf99 = extern_kernels.convolution(buf98, arg165_1, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf99, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg165_1
    del arg166_1
    del buf98
    buf100 = buf99; del buf99  # reuse
    cpp_fused_silu_36(c_void_p(buf100.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.silu]
    buf101 = extern_kernels.convolution(buf100, arg167_1, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf101, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg167_1
    del arg168_1
    del buf100
    buf102 = reinterpret_tensor(buf90, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf90  # reuse
    buf103 = buf95; del buf95  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_37(c_void_p(buf96.data_ptr()), c_void_p(buf101.data_ptr()), c_void_p(buf102.data_ptr()), c_void_p(buf103.data_ptr()))
    del buf96
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf103, arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg169_1
    buf105 = buf103; del buf103  # reuse
    cpp_fused_convolution_38(c_void_p(buf102.data_ptr()), c_void_p(buf105.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg170_1
    del buf105
    buf107 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_39(c_void_p(buf107.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg345_1
    del arg346_1
    del arg40_1
    del arg41_1
    del buf104
    del buf106
    # Source Nodes: [cat_67, shortcut_7, x_127, x_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf108 = extern_kernels.convolution(buf107, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf108, (8, 336, 28, 28), (263424, 1, 9408, 336))
    del arg171_1
    del buf107
    buf109 = buf108; del buf108  # reuse
    buf110 = buf102; del buf102  # reuse
    buf111 = empty_strided((8, 112, 28, 28), (87808, 1, 3136, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_40(c_void_p(buf109.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf111.data_ptr()))
    del arg347_1
    del arg348_1
    del arg42_1
    del arg43_1
    del buf109
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_0], Original ATen: [aten.convolution]
    buf112 = extern_kernels.convolution(buf111, arg172_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf112, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg172_1
    buf113 = buf111; del buf111  # reuse
    cpp_fused_convolution_41(c_void_p(buf110.data_ptr()), c_void_p(buf113.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_1], Original ATen: [aten.convolution]
    buf114 = extern_kernels.convolution(buf113, arg173_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf114, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg173_1
    buf115 = buf113; del buf113  # reuse
    cpp_fused_convolution_42(c_void_p(buf110.data_ptr()), c_void_p(buf115.data_ptr()))
    del buf110
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___conv_dw_2], Original ATen: [aten.convolution]
    buf116 = extern_kernels.convolution(buf115, arg174_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf116, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg174_1
    del buf115
    buf117 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    buf118 = reinterpret_tensor(buf101, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf101  # reuse
    buf119 = reinterpret_tensor(buf118, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf118  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_43(c_void_p(buf119.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf114.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf117.data_ptr()))
    del arg349_1
    del arg350_1
    del arg44_1
    del arg45_1
    del buf112
    del buf114
    del buf116
    # Source Nodes: [x_142, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf120 = extern_kernels.convolution(buf119, arg175_1, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf120, (8, 14, 1, 1), (14, 1, 14, 14))
    del arg175_1
    del arg176_1
    del buf119
    buf121 = buf120; del buf120  # reuse
    cpp_fused_silu_44(c_void_p(buf121.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.silu]
    buf122 = extern_kernels.convolution(buf121, arg177_1, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf122, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg177_1
    del arg178_1
    del buf121
    buf123 = buf117; del buf117  # reuse
    cpp_fused_mul_sigmoid_silu_45(c_void_p(buf123.data_ptr()), c_void_p(buf122.data_ptr()))
    del buf122
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_142, x_143, x_144], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf124 = extern_kernels.convolution(buf123, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf124, (8, 104, 14, 14), (20384, 1, 1456, 104))
    del arg179_1
    del buf123
    buf125 = buf124; del buf124  # reuse
    cpp_fused__native_batch_norm_legit_no_training_46(c_void_p(buf125.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg351_1
    del arg352_1
    del arg46_1
    del arg47_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf126 = extern_kernels.convolution(reinterpret_tensor(buf125, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf126, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg180_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(reinterpret_tensor(buf125, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg181_1
    buf128 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    buf129 = empty((8, 624, 14, 14), device='cpu', dtype=torch.float32)
    buf130 = empty_strided((8, 156, 14, 14), (30576, 1, 2184, 156), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_47(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()))
    del arg353_1
    del arg354_1
    del arg48_1
    del arg49_1
    del buf126
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf131 = extern_kernels.convolution(buf130, arg182_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf131, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg182_1
    buf132 = buf130; del buf130  # reuse
    cpp_fused_convolution_48(c_void_p(buf129.data_ptr()), c_void_p(buf132.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf133 = extern_kernels.convolution(buf132, arg183_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf133, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg183_1
    buf134 = buf132; del buf132  # reuse
    cpp_fused_convolution_49(c_void_p(buf129.data_ptr()), c_void_p(buf134.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf135 = extern_kernels.convolution(buf134, arg184_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf135, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg184_1
    buf136 = buf134; del buf134  # reuse
    cpp_fused_convolution_50(c_void_p(buf129.data_ptr()), c_void_p(buf136.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf137 = extern_kernels.convolution(buf136, arg185_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf137, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg185_1
    del buf136
    buf138 = reinterpret_tensor(buf129, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf129  # reuse
    buf139 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cpu', dtype=torch.float32)
    buf140 = reinterpret_tensor(buf139, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf139  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_51(c_void_p(buf140.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf135.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf138.data_ptr()))
    del arg355_1
    del arg356_1
    del arg50_1
    del arg51_1
    del buf131
    del buf133
    del buf135
    # Source Nodes: [x_160, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf141 = extern_kernels.convolution(buf140, arg186_1, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf141, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg186_1
    del arg187_1
    del buf140
    buf142 = buf141; del buf141  # reuse
    cpp_fused_silu_52(c_void_p(buf142.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.silu]
    buf143 = extern_kernels.convolution(buf142, arg188_1, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf143, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg188_1
    del arg189_1
    del buf142
    buf144 = reinterpret_tensor(buf128, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf128  # reuse
    buf145 = buf127; del buf127  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_53(c_void_p(buf138.data_ptr()), c_void_p(buf143.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf146 = extern_kernels.convolution(buf145, arg190_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf146, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg190_1
    buf147 = buf145; del buf145  # reuse
    cpp_fused_convolution_54(c_void_p(buf144.data_ptr()), c_void_p(buf147.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf148 = extern_kernels.convolution(buf147, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf148, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg191_1
    del buf147
    buf149 = buf125; del buf125  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_55(c_void_p(buf149.data_ptr()), c_void_p(buf146.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg357_1
    del arg358_1
    del arg52_1
    del arg53_1
    del buf146
    del buf148
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf150 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf150, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg192_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(reinterpret_tensor(buf149, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg193_1
    buf152 = reinterpret_tensor(buf144, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf144  # reuse
    buf153 = reinterpret_tensor(buf138, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf138  # reuse
    buf154 = buf137; del buf137  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_56(c_void_p(buf150.data_ptr()), c_void_p(buf151.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()))
    del arg359_1
    del arg360_1
    del arg54_1
    del arg55_1
    del buf150
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf155 = extern_kernels.convolution(buf154, arg194_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf155, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg194_1
    buf156 = buf154; del buf154  # reuse
    cpp_fused_convolution_57(c_void_p(buf153.data_ptr()), c_void_p(buf156.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf157 = extern_kernels.convolution(buf156, arg195_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf157, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg195_1
    buf158 = buf156; del buf156  # reuse
    cpp_fused_convolution_58(c_void_p(buf153.data_ptr()), c_void_p(buf158.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf159 = extern_kernels.convolution(buf158, arg196_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf159, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg196_1
    buf160 = buf158; del buf158  # reuse
    cpp_fused_convolution_59(c_void_p(buf153.data_ptr()), c_void_p(buf160.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf161 = extern_kernels.convolution(buf160, arg197_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf161, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg197_1
    del buf160
    buf162 = reinterpret_tensor(buf153, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf153  # reuse
    buf163 = reinterpret_tensor(buf143, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf143  # reuse
    buf164 = reinterpret_tensor(buf163, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf163  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_60(c_void_p(buf164.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(buf162.data_ptr()))
    del arg361_1
    del arg362_1
    del arg56_1
    del arg57_1
    del buf155
    del buf157
    del buf159
    # Source Nodes: [x_180, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf165 = extern_kernels.convolution(buf164, arg198_1, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf165, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg198_1
    del arg199_1
    del buf164
    buf166 = buf165; del buf165  # reuse
    cpp_fused_silu_61(c_void_p(buf166.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.silu]
    buf167 = extern_kernels.convolution(buf166, arg200_1, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf167, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg200_1
    del arg201_1
    del buf166
    buf168 = reinterpret_tensor(buf152, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf152  # reuse
    buf169 = buf151; del buf151  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_62(c_void_p(buf162.data_ptr()), c_void_p(buf167.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf170 = extern_kernels.convolution(buf169, arg202_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf170, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg202_1
    buf171 = buf169; del buf169  # reuse
    cpp_fused_convolution_63(c_void_p(buf168.data_ptr()), c_void_p(buf171.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf172 = extern_kernels.convolution(buf171, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf172, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg203_1
    del buf171
    buf173 = buf149; del buf149  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_64(c_void_p(buf173.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg363_1
    del arg364_1
    del arg58_1
    del arg59_1
    del buf170
    del buf172
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf174 = extern_kernels.convolution(reinterpret_tensor(buf173, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg204_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf174, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg204_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(reinterpret_tensor(buf173, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg205_1
    buf176 = reinterpret_tensor(buf168, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf168  # reuse
    buf177 = reinterpret_tensor(buf162, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf162  # reuse
    buf178 = buf161; del buf161  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_65(c_void_p(buf174.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()))
    del arg365_1
    del arg366_1
    del arg60_1
    del arg61_1
    del buf174
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf179 = extern_kernels.convolution(buf178, arg206_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf179, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg206_1
    buf180 = buf178; del buf178  # reuse
    cpp_fused_convolution_66(c_void_p(buf177.data_ptr()), c_void_p(buf180.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf181 = extern_kernels.convolution(buf180, arg207_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf181, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg207_1
    buf182 = buf180; del buf180  # reuse
    cpp_fused_convolution_67(c_void_p(buf177.data_ptr()), c_void_p(buf182.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf183 = extern_kernels.convolution(buf182, arg208_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf183, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg208_1
    buf184 = buf182; del buf182  # reuse
    cpp_fused_convolution_68(c_void_p(buf177.data_ptr()), c_void_p(buf184.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf185 = extern_kernels.convolution(buf184, arg209_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf185, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg209_1
    del buf184
    buf186 = reinterpret_tensor(buf177, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf177  # reuse
    buf187 = reinterpret_tensor(buf167, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf167  # reuse
    buf188 = reinterpret_tensor(buf187, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf187  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_69(c_void_p(buf188.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf181.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(buf186.data_ptr()))
    del arg367_1
    del arg368_1
    del arg62_1
    del arg63_1
    del buf179
    del buf181
    del buf183
    del buf185
    # Source Nodes: [x_200, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf189 = extern_kernels.convolution(buf188, arg210_1, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf189, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg210_1
    del arg211_1
    del buf188
    buf190 = buf189; del buf189  # reuse
    cpp_fused_silu_70(c_void_p(buf190.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.silu]
    buf191 = extern_kernels.convolution(buf190, arg212_1, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf191, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg212_1
    del arg213_1
    del buf190
    buf192 = reinterpret_tensor(buf176, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf176  # reuse
    buf193 = buf175; del buf175  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_71(c_void_p(buf186.data_ptr()), c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()))
    del buf186
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf194 = extern_kernels.convolution(buf193, arg214_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf194, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg214_1
    buf195 = buf193; del buf193  # reuse
    cpp_fused_convolution_72(c_void_p(buf192.data_ptr()), c_void_p(buf195.data_ptr()))
    del buf192
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf196 = extern_kernels.convolution(buf195, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf196, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg215_1
    del buf195
    buf197 = buf173; del buf173  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_73(c_void_p(buf197.data_ptr()), c_void_p(buf194.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg369_1
    del arg370_1
    del arg64_1
    del arg65_1
    del buf194
    del buf196
    # Source Nodes: [cat_57, shortcut_11, x_204, x_209], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf198 = extern_kernels.convolution(buf197, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf198, (8, 624, 14, 14), (122304, 1, 8736, 624))
    del arg216_1
    del buf197
    buf199 = buf198; del buf198  # reuse
    buf200 = buf199; del buf199  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_74(c_void_p(buf200.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()))
    del arg371_1
    del arg372_1
    del arg66_1
    del arg67_1
    # Source Nodes: [x_213, x_214], Original ATen: [aten.convolution, aten.silu]
    buf201 = extern_kernels.convolution(buf200, arg217_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
    assert_size_stride(buf201, (8, 624, 14, 14), (122304, 1, 8736, 624))
    del arg217_1
    del buf200
    buf202 = buf201; del buf201  # reuse
    buf203 = reinterpret_tensor(buf191, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf191  # reuse
    buf204 = reinterpret_tensor(buf203, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf203  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_75(c_void_p(buf202.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()))
    del arg373_1
    del arg374_1
    del arg68_1
    del arg69_1
    # Source Nodes: [x_218, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf205 = extern_kernels.convolution(buf204, arg218_1, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf205, (8, 52, 1, 1), (52, 1, 52, 52))
    del arg218_1
    del arg219_1
    del buf204
    buf206 = buf205; del buf205  # reuse
    cpp_fused_silu_76(c_void_p(buf206.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.silu]
    buf207 = extern_kernels.convolution(buf206, arg220_1, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf207, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg220_1
    del arg221_1
    del buf206
    buf208 = buf202; del buf202  # reuse
    cpp_fused_mul_sigmoid_silu_77(c_void_p(buf208.data_ptr()), c_void_p(buf207.data_ptr()))
    del buf207
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_218, x_219, x_220], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf209 = extern_kernels.convolution(buf208, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf209, (8, 160, 14, 14), (31360, 1, 2240, 160))
    del arg222_1
    del buf208
    buf210 = buf209; del buf209  # reuse
    cpp_fused__native_batch_norm_legit_no_training_78(c_void_p(buf210.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg375_1
    del arg376_1
    del arg70_1
    del arg71_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf211 = extern_kernels.convolution(reinterpret_tensor(buf210, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf211, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg223_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(reinterpret_tensor(buf210, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf212, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg224_1
    buf213 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf214 = empty((8, 480, 14, 14), device='cpu', dtype=torch.float32)
    buf215 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_79(c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()))
    del arg377_1
    del arg378_1
    del arg72_1
    del arg73_1
    del buf211
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf216 = extern_kernels.convolution(buf215, arg225_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf216, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg225_1
    buf217 = buf215; del buf215  # reuse
    cpp_fused_convolution_80(c_void_p(buf214.data_ptr()), c_void_p(buf217.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf218 = extern_kernels.convolution(buf217, arg226_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf218, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg226_1
    buf219 = buf217; del buf217  # reuse
    cpp_fused_convolution_81(c_void_p(buf214.data_ptr()), c_void_p(buf219.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf220 = extern_kernels.convolution(buf219, arg227_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf220, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg227_1
    buf221 = buf219; del buf219  # reuse
    cpp_fused_convolution_82(c_void_p(buf214.data_ptr()), c_void_p(buf221.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf222 = extern_kernels.convolution(buf221, arg228_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf222, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg228_1
    del buf221
    buf223 = reinterpret_tensor(buf214, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf214  # reuse
    buf224 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf225 = reinterpret_tensor(buf224, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf224  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_83(c_void_p(buf225.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf222.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(buf223.data_ptr()))
    del arg379_1
    del arg380_1
    del arg74_1
    del arg75_1
    del buf216
    del buf218
    del buf220
    # Source Nodes: [x_236, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf226 = extern_kernels.convolution(buf225, arg229_1, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf226, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg229_1
    del arg230_1
    del buf225
    buf227 = buf226; del buf226  # reuse
    cpp_fused_silu_84(c_void_p(buf227.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.silu]
    buf228 = extern_kernels.convolution(buf227, arg231_1, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf228, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg231_1
    del arg232_1
    del buf227
    buf229 = reinterpret_tensor(buf213, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf213  # reuse
    buf230 = buf212; del buf212  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_85(c_void_p(buf223.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf231 = extern_kernels.convolution(buf230, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf231, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg233_1
    buf232 = buf230; del buf230  # reuse
    cpp_fused_convolution_86(c_void_p(buf229.data_ptr()), c_void_p(buf232.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf233 = extern_kernels.convolution(buf232, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf233, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg234_1
    del buf232
    buf234 = buf210; del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_87(c_void_p(buf234.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg381_1
    del arg382_1
    del arg76_1
    del arg77_1
    del buf231
    del buf233
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf235 = extern_kernels.convolution(reinterpret_tensor(buf234, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf235, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg235_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf236 = extern_kernels.convolution(reinterpret_tensor(buf234, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf236, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg236_1
    buf237 = reinterpret_tensor(buf229, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf229  # reuse
    buf238 = reinterpret_tensor(buf223, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf223  # reuse
    buf239 = buf222; del buf222  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_88(c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()))
    del arg383_1
    del arg384_1
    del arg78_1
    del arg79_1
    del buf235
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf240 = extern_kernels.convolution(buf239, arg237_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf240, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg237_1
    buf241 = buf239; del buf239  # reuse
    cpp_fused_convolution_89(c_void_p(buf238.data_ptr()), c_void_p(buf241.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf242 = extern_kernels.convolution(buf241, arg238_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf242, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg238_1
    buf243 = buf241; del buf241  # reuse
    cpp_fused_convolution_90(c_void_p(buf238.data_ptr()), c_void_p(buf243.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf244 = extern_kernels.convolution(buf243, arg239_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf244, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg239_1
    buf245 = buf243; del buf243  # reuse
    cpp_fused_convolution_91(c_void_p(buf238.data_ptr()), c_void_p(buf245.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf246 = extern_kernels.convolution(buf245, arg240_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf246, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg240_1
    del buf245
    buf247 = reinterpret_tensor(buf238, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf238  # reuse
    buf248 = reinterpret_tensor(buf228, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf228  # reuse
    buf249 = reinterpret_tensor(buf248, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf248  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_92(c_void_p(buf249.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf242.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(buf247.data_ptr()))
    del arg385_1
    del arg386_1
    del arg80_1
    del arg81_1
    del buf240
    del buf242
    del buf244
    # Source Nodes: [x_256, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf250 = extern_kernels.convolution(buf249, arg241_1, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf250, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg241_1
    del arg242_1
    del buf249
    buf251 = buf250; del buf250  # reuse
    cpp_fused_silu_93(c_void_p(buf251.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.silu]
    buf252 = extern_kernels.convolution(buf251, arg243_1, arg244_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf252, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg243_1
    del arg244_1
    del buf251
    buf253 = reinterpret_tensor(buf237, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf237  # reuse
    buf254 = buf236; del buf236  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_94(c_void_p(buf247.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf255 = extern_kernels.convolution(buf254, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf255, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg245_1
    buf256 = buf254; del buf254  # reuse
    cpp_fused_convolution_95(c_void_p(buf253.data_ptr()), c_void_p(buf256.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf257 = extern_kernels.convolution(buf256, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf257, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg246_1
    del buf256
    buf258 = buf234; del buf234  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_96(c_void_p(buf258.data_ptr()), c_void_p(buf255.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg387_1
    del arg388_1
    del arg82_1
    del arg83_1
    del buf255
    del buf257
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf259 = extern_kernels.convolution(reinterpret_tensor(buf258, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf259, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg247_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(reinterpret_tensor(buf258, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf260, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg248_1
    buf261 = reinterpret_tensor(buf253, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf253  # reuse
    buf262 = reinterpret_tensor(buf247, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf247  # reuse
    buf263 = buf246; del buf246  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_97(c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()))
    del arg389_1
    del arg390_1
    del arg84_1
    del arg85_1
    del buf259
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf264 = extern_kernels.convolution(buf263, arg249_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf264, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg249_1
    buf265 = buf263; del buf263  # reuse
    cpp_fused_convolution_98(c_void_p(buf262.data_ptr()), c_void_p(buf265.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf266 = extern_kernels.convolution(buf265, arg250_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf266, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg250_1
    buf267 = buf265; del buf265  # reuse
    cpp_fused_convolution_99(c_void_p(buf262.data_ptr()), c_void_p(buf267.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf268 = extern_kernels.convolution(buf267, arg251_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf268, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg251_1
    buf269 = buf267; del buf267  # reuse
    cpp_fused_convolution_100(c_void_p(buf262.data_ptr()), c_void_p(buf269.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf270 = extern_kernels.convolution(buf269, arg252_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf270, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg252_1
    del buf269
    buf271 = reinterpret_tensor(buf262, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf262  # reuse
    buf272 = reinterpret_tensor(buf252, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf252  # reuse
    buf273 = reinterpret_tensor(buf272, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf272  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_101(c_void_p(buf273.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf270.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(buf271.data_ptr()))
    del arg391_1
    del arg392_1
    del arg86_1
    del arg87_1
    del buf264
    del buf266
    del buf268
    del buf270
    # Source Nodes: [x_276, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf274 = extern_kernels.convolution(buf273, arg253_1, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf274, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg253_1
    del arg254_1
    del buf273
    buf275 = buf274; del buf274  # reuse
    cpp_fused_silu_102(c_void_p(buf275.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.silu]
    buf276 = extern_kernels.convolution(buf275, arg255_1, arg256_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf276, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg255_1
    del arg256_1
    del buf275
    buf277 = reinterpret_tensor(buf261, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf261  # reuse
    buf278 = buf260; del buf260  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_103(c_void_p(buf271.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del buf271
    del buf276
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf279 = extern_kernels.convolution(buf278, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf279, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg257_1
    buf280 = buf278; del buf278  # reuse
    cpp_fused_convolution_104(c_void_p(buf277.data_ptr()), c_void_p(buf280.data_ptr()))
    del buf277
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf281 = extern_kernels.convolution(buf280, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf281, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg258_1
    buf282 = buf258; del buf258  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_105(c_void_p(buf282.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(buf281.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg393_1
    del arg394_1
    del arg88_1
    del arg89_1
    del buf279
    del buf281
    # Source Nodes: [cat_48, shortcut_15, x_280, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf283 = extern_kernels.convolution(buf282, arg259_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf283, (8, 960, 14, 14), (188160, 1, 13440, 960))
    del arg259_1
    del buf282
    buf284 = buf283; del buf283  # reuse
    buf285 = reinterpret_tensor(buf45, (8, 960, 14, 14), (188160, 196, 14, 1), 0); del buf45  # reuse
    buf286 = buf280; del buf280  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_106(c_void_p(buf284.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del arg395_1
    del arg396_1
    del arg90_1
    del arg91_1
    del buf284
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_0], Original ATen: [aten.convolution]
    buf287 = extern_kernels.convolution(buf286, arg260_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf287, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg260_1
    buf288 = buf286; del buf286  # reuse
    cpp_fused_convolution_107(c_void_p(buf285.data_ptr()), c_void_p(buf288.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_1], Original ATen: [aten.convolution]
    buf289 = extern_kernels.convolution(buf288, arg261_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf289, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg261_1
    buf290 = buf288; del buf288  # reuse
    cpp_fused_convolution_108(c_void_p(buf285.data_ptr()), c_void_p(buf290.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_2], Original ATen: [aten.convolution]
    buf291 = extern_kernels.convolution(buf290, arg262_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf291, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg262_1
    buf292 = buf290; del buf290  # reuse
    cpp_fused_convolution_109(c_void_p(buf285.data_ptr()), c_void_p(buf292.data_ptr()))
    del buf285
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___conv_dw_3], Original ATen: [aten.convolution]
    buf293 = extern_kernels.convolution(buf292, arg263_1, stride=(2, 2), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf293, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg263_1
    buf294 = reinterpret_tensor(buf292, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf292  # reuse
    buf295 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf296 = reinterpret_tensor(buf295, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf295  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_110(c_void_p(buf296.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg397_1
    del arg398_1
    del arg92_1
    del arg93_1
    del buf287
    del buf289
    del buf291
    del buf293
    # Source Nodes: [x_295, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf297 = extern_kernels.convolution(buf296, arg264_1, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf297, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg264_1
    del arg265_1
    del buf296
    buf298 = buf297; del buf297  # reuse
    cpp_fused_silu_111(c_void_p(buf298.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.silu]
    buf299 = extern_kernels.convolution(buf298, arg266_1, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf299, (8, 960, 1, 1), (960, 1, 960, 960))
    del arg266_1
    del arg267_1
    del buf298
    buf300 = buf294; del buf294  # reuse
    cpp_fused_mul_sigmoid_silu_112(c_void_p(buf300.data_ptr()), c_void_p(buf299.data_ptr()))
    del buf299
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_295, x_296, x_297], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf301 = extern_kernels.convolution(buf300, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf301, (8, 264, 7, 7), (12936, 1, 1848, 264))
    del arg268_1
    del buf300
    buf302 = buf301; del buf301  # reuse
    cpp_fused__native_batch_norm_legit_no_training_113(c_void_p(buf302.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg399_1
    del arg400_1
    del arg94_1
    del arg95_1
    # Source Nodes: [x_302], Original ATen: [aten.convolution]
    buf303 = extern_kernels.convolution(buf302, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf303, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg269_1
    buf304 = buf303; del buf303  # reuse
    buf305 = empty((8, 1584, 7, 7), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 396, 7, 7), (19404, 1, 2772, 396), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_114(c_void_p(buf304.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg401_1
    del arg402_1
    del arg96_1
    del arg97_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf307 = extern_kernels.convolution(buf306, arg270_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf307, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg270_1
    buf308 = buf306; del buf306  # reuse
    cpp_fused_convolution_115(c_void_p(buf305.data_ptr()), c_void_p(buf308.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf309 = extern_kernels.convolution(buf308, arg271_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf309, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg271_1
    buf310 = buf308; del buf308  # reuse
    cpp_fused_convolution_116(c_void_p(buf305.data_ptr()), c_void_p(buf310.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf311 = extern_kernels.convolution(buf310, arg272_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf311, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg272_1
    buf312 = buf310; del buf310  # reuse
    cpp_fused_convolution_117(c_void_p(buf305.data_ptr()), c_void_p(buf312.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf313 = extern_kernels.convolution(buf312, arg273_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf313, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg273_1
    del buf312
    buf314 = reinterpret_tensor(buf305, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf305  # reuse
    buf315 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cpu', dtype=torch.float32)
    buf316 = reinterpret_tensor(buf315, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf315  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_118(c_void_p(buf316.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg403_1
    del arg404_1
    del arg98_1
    del arg99_1
    del buf307
    del buf309
    del buf311
    # Source Nodes: [x_312, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf317 = extern_kernels.convolution(buf316, arg274_1, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf317, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg274_1
    del arg275_1
    del buf316
    buf318 = buf317; del buf317  # reuse
    cpp_fused_silu_119(c_void_p(buf318.data_ptr()))
    # Source Nodes: [x_se_54, x_se_55], Original ATen: [aten.convolution, aten.silu]
    buf319 = extern_kernels.convolution(buf318, arg276_1, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf319, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg276_1
    del arg277_1
    del buf318
    buf320 = reinterpret_tensor(buf304, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf304  # reuse
    buf321 = empty_strided((8, 792, 7, 7), (38808, 1, 5544, 792), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_mul_sigmoid_silu_120(c_void_p(buf314.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del buf314
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf322 = extern_kernels.convolution(buf321, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf322, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg278_1
    buf323 = buf321; del buf321  # reuse
    cpp_fused_convolution_121(c_void_p(buf320.data_ptr()), c_void_p(buf323.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf324 = extern_kernels.convolution(buf323, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf324, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg279_1
    buf325 = buf302; del buf302  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_122(c_void_p(buf325.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg405_1
    del arg406_1
    del buf322
    del buf324
    # Source Nodes: [x_321], Original ATen: [aten.convolution]
    buf326 = extern_kernels.convolution(buf325, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf326, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg280_1
    buf327 = buf326; del buf326  # reuse
    buf328 = buf320; del buf320  # reuse
    buf329 = buf313; del buf313  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_123(c_void_p(buf327.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    del arg102_1
    del arg103_1
    del arg407_1
    del arg408_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf330 = extern_kernels.convolution(buf329, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf330, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg281_1
    buf331 = buf329; del buf329  # reuse
    cpp_fused_convolution_124(c_void_p(buf328.data_ptr()), c_void_p(buf331.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf332 = extern_kernels.convolution(buf331, arg282_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf332, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg282_1
    buf333 = buf331; del buf331  # reuse
    cpp_fused_convolution_125(c_void_p(buf328.data_ptr()), c_void_p(buf333.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf334 = extern_kernels.convolution(buf333, arg283_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf334, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg283_1
    buf335 = buf333; del buf333  # reuse
    cpp_fused_convolution_126(c_void_p(buf328.data_ptr()), c_void_p(buf335.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf336 = extern_kernels.convolution(buf335, arg284_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf336, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg284_1
    del buf335
    buf337 = reinterpret_tensor(buf328, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf328  # reuse
    buf338 = reinterpret_tensor(buf319, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf319  # reuse
    buf339 = reinterpret_tensor(buf338, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf338  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_127(c_void_p(buf339.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(buf337.data_ptr()))
    del arg104_1
    del arg105_1
    del arg409_1
    del arg410_1
    del buf330
    del buf332
    del buf334
    # Source Nodes: [x_331, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf340 = extern_kernels.convolution(buf339, arg285_1, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf340, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg285_1
    del arg286_1
    del buf339
    buf341 = buf340; del buf340  # reuse
    cpp_fused_silu_128(c_void_p(buf341.data_ptr()))
    # Source Nodes: [x_se_58, x_se_59], Original ATen: [aten.convolution, aten.silu]
    buf342 = extern_kernels.convolution(buf341, arg287_1, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf342, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg287_1
    del arg288_1
    del buf341
    buf343 = reinterpret_tensor(buf327, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf327  # reuse
    buf344 = buf323; del buf323  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_129(c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del buf337
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf345 = extern_kernels.convolution(buf344, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf345, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg289_1
    buf346 = buf344; del buf344  # reuse
    cpp_fused_convolution_130(c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf347 = extern_kernels.convolution(buf346, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf347, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg290_1
    buf348 = buf325; del buf325  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_131(c_void_p(buf348.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg411_1
    del arg412_1
    del buf345
    del buf347
    # Source Nodes: [x_340], Original ATen: [aten.convolution]
    buf349 = extern_kernels.convolution(buf348, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf349, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg291_1
    buf350 = buf349; del buf349  # reuse
    buf351 = buf343; del buf343  # reuse
    buf352 = buf336; del buf336  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_132(c_void_p(buf350.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del arg108_1
    del arg109_1
    del arg413_1
    del arg414_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf353 = extern_kernels.convolution(buf352, arg292_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf353, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg292_1
    buf354 = buf352; del buf352  # reuse
    cpp_fused_convolution_133(c_void_p(buf351.data_ptr()), c_void_p(buf354.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf355 = extern_kernels.convolution(buf354, arg293_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf355, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg293_1
    buf356 = buf354; del buf354  # reuse
    cpp_fused_convolution_134(c_void_p(buf351.data_ptr()), c_void_p(buf356.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, arg294_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf357, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg294_1
    buf358 = buf356; del buf356  # reuse
    cpp_fused_convolution_135(c_void_p(buf351.data_ptr()), c_void_p(buf358.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf359 = extern_kernels.convolution(buf358, arg295_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf359, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg295_1
    del buf358
    buf360 = reinterpret_tensor(buf351, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf351  # reuse
    buf361 = reinterpret_tensor(buf342, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf342  # reuse
    buf362 = reinterpret_tensor(buf361, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf361  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_136(c_void_p(buf362.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf360.data_ptr()))
    del arg110_1
    del arg111_1
    del arg415_1
    del arg416_1
    del buf353
    del buf355
    del buf357
    del buf359
    # Source Nodes: [x_350, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf363 = extern_kernels.convolution(buf362, arg296_1, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf363, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg296_1
    del arg297_1
    del buf362
    buf364 = buf363; del buf363  # reuse
    cpp_fused_silu_137(c_void_p(buf364.data_ptr()))
    # Source Nodes: [x_se_62, x_se_63], Original ATen: [aten.convolution, aten.silu]
    buf365 = extern_kernels.convolution(buf364, arg298_1, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf365, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg298_1
    del arg299_1
    del buf364
    buf366 = reinterpret_tensor(buf350, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf350  # reuse
    buf367 = buf346; del buf346  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_138(c_void_p(buf360.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del buf360
    del buf365
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf368 = extern_kernels.convolution(buf367, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf368, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg300_1
    buf369 = buf367; del buf367  # reuse
    cpp_fused_convolution_139(c_void_p(buf366.data_ptr()), c_void_p(buf369.data_ptr()))
    del buf366
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf370 = extern_kernels.convolution(buf369, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf370, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg301_1
    del buf369
    buf371 = buf348; del buf348  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_140(c_void_p(buf371.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg417_1
    del arg418_1
    del buf368
    del buf370
    # Source Nodes: [cat_41, x_354, x_359, x_360], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf372 = extern_kernels.convolution(buf371, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf372, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del arg302_1
    del buf371
    buf373 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf374 = reinterpret_tensor(buf373, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf373  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_141(c_void_p(buf374.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()))
    del arg114_1
    del arg115_1
    del arg419_1
    del arg420_1
    del buf372
    buf375 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_369], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf374, (8, 1536), (1536, 1), 0), reinterpret_tensor(arg303_1, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf375)
    del arg303_1
    del arg304_1
    return (buf375, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((14, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((52, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((80, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((960, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((264, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((132, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((132, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg291_1 = rand_strided((1584, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg292_1 = rand_strided((396, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg293_1 = rand_strided((396, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg294_1 = rand_strided((396, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg295_1 = rand_strided((396, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg296_1 = rand_strided((132, 1584, 1, 1), (1584, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg297_1 = rand_strided((132, ), (1, ), device='cpu', dtype=torch.float32)
    arg298_1 = rand_strided((1584, 132, 1, 1), (132, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg299_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg300_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg301_1 = rand_strided((132, 792, 1, 1), (792, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg302_1 = rand_strided((1536, 264, 1, 1), (264, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg303_1 = rand_strided((1000, 1536), (1536, 1), device='cpu', dtype=torch.float32)
    arg304_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg305_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg306_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg307_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg308_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg309_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg310_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg311_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg312_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg313_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg314_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg315_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg316_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg317_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg318_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg319_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg320_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg321_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg322_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg323_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg324_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg325_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg326_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg327_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg328_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg329_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg330_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg331_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg332_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg333_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg334_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg335_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg336_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg337_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg338_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg339_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg340_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg341_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg342_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg343_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg344_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg345_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg346_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg347_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg348_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg349_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg350_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg351_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg352_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg353_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg354_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg355_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg356_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg357_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg358_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg359_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg360_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg361_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg362_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg363_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg364_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg365_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg366_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg367_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg368_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg369_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg370_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg371_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg372_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg373_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg374_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg375_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg376_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg377_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg378_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg379_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg380_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg381_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg382_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg383_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg384_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg385_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg386_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg387_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg388_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg389_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg390_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg391_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg392_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg393_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg394_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg395_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg396_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg397_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg398_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg399_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg400_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg401_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg402_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg403_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg404_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg405_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg406_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg407_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg408_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg409_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg410_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg411_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg412_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg413_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg414_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg415_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg416_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg417_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg418_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg419_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg420_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg421_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mixnet_l', benchmark_compiled_module)
