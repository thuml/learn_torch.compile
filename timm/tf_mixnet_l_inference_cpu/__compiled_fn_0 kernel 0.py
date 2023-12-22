
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


cpp_fused_constant_pad_nd_convolution_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(225L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(225L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(224);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<long>(x3);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_ptr0[static_cast<long>(x3 + (224L*x2) + (50176L*x1) + (150528L*x0))];
                                return tmp7;
                            }
                            ;
                            auto tmp8 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            out_ptr0[static_cast<long>(x1 + (3L*x3) + (675L*x2) + (151875L*x0))] = tmp8;
                        }
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


cpp_fused__native_batch_norm_legit_no_training_cat_constant_pad_nd_relu_5 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(113L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(113L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(112);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(out_ptr0 + static_cast<long>(x3 + (192L*x2) + (21504L*x1) + (2408448L*x0)), to_float_mask(tmp5));
                                return tmp7;
                            }
                            ;
                            auto tmp8 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp8.store(out_ptr1 + static_cast<long>(x3 + (64L*x2) + (7232L*x1) + (817216L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_6 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(115L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(115L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-1L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-21632L) + x3 + (192L*x2) + (21504L*x1) + (2408448L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp13.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (7360L*x1) + (846400L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_7 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(117L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(117L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(64L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(112);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-43264L) + x3 + (192L*x2) + (21504L*x1) + (2408448L*x0)), to_float_mask(tmp10));
                                return tmp12;
                            }
                            ;
                            auto tmp13 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp13.store(out_ptr0 + static_cast<long>(x3 + (64L*x2) + (7488L*x1) + (876096L*x0)));
                        }
                    }
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
                    auto tmp26 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_9 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_relu_10 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.cpp('''
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
                    tmp17.store(in_out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_cat_12 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_13 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(57L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(57L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(56);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (240L*x2) + (13440L*x1) + (752640L*x0)), to_float_mask(tmp5));
                                auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                                auto tmp9 = tmp7 * tmp8;
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp10.store(out_ptr0 + static_cast<long>(x3 + (60L*x2) + (3420L*x1) + (194940L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(56L); x3<static_cast<long>(60L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x1);
                            auto tmp1 = static_cast<long>(56);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<long>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = in_out_ptr0[static_cast<long>(x3 + (240L*x2) + (13440L*x1) + (752640L*x0))];
                                auto tmp8 = decltype(tmp7)(1) / (decltype(tmp7)(1) + std::exp(-tmp7));
                                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                                return tmp9;
                            }
                            ;
                            auto tmp10 = tmp5 ? tmp6() : static_cast<decltype(tmp6())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (60L*x2) + (3420L*x1) + (194940L*x0))] = tmp10;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_14 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(59L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(59L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-1L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-13620L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (60L*x2) + (3540L*x1) + (208860L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(56L); x3<static_cast<long>(60L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-1L) + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>((-1L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-13620L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0))];
                                auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (60L*x2) + (3540L*x1) + (208860L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(61L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(61L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-27240L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (60L*x2) + (3660L*x1) + (223260L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(56L); x3<static_cast<long>(60L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-2L) + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-27240L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0))];
                                auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (60L*x2) + (3660L*x1) + (223260L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_16 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(63L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(63L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(56L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-3L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-3L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-40860L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (60L*x2) + (3780L*x1) + (238140L*x0)));
                        }
                        #pragma omp simd simdlen(4) 
                        for(long x3=static_cast<long>(56L); x3<static_cast<long>(60L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>((-3L) + x1);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(56);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<long>((-3L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = in_ptr0[static_cast<long>((-40860L) + x3 + (240L*x2) + (13440L*x1) + (752640L*x0))];
                                auto tmp13 = decltype(tmp12)(1) / (decltype(tmp12)(1) + std::exp(-tmp12));
                                auto tmp14 = decltype(tmp12)(tmp12 * tmp13);
                                return tmp14;
                            }
                            ;
                            auto tmp15 = tmp10 ? tmp11() : static_cast<decltype(tmp11())>(0.0);
                            out_ptr0[static_cast<long>(x3 + (60L*x2) + (3780L*x1) + (238140L*x0))] = tmp15;
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_17 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_18 = async_compile.cpp('''
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


cpp_fused_mul_sigmoid_silu_19 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_20 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (56L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_21 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_22 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_23 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_silu_24 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_25 = async_compile.cpp('''
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


cpp_fused_convolution_26 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_27 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_28 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_29 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_30 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_silu_31 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_32 = async_compile.cpp('''
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


cpp_fused_convolution_33 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_34 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_35 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_36 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_37 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_silu_38 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_39 = async_compile.cpp('''
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


cpp_fused_convolution_40 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_41 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_42 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(336L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (336L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(29L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(29L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(112L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(28);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (336L*x2) + (9408L*x1) + (263424L*x0)), to_float_mask(tmp5));
                                auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                                auto tmp9 = tmp7 * tmp8;
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp10.store(out_ptr0 + static_cast<long>(x3 + (112L*x2) + (3248L*x1) + (94192L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_43 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(31L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(31L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(112L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-1L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-9632L) + x3 + (336L*x2) + (9408L*x1) + (263424L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (112L*x2) + (3472L*x1) + (107632L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_44 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(33L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(33L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(112L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(28);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-19264L) + x3 + (336L*x2) + (9408L*x1) + (263424L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (112L*x2) + (3696L*x1) + (121968L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_45 = async_compile.cpp('''
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
                    auto tmp26 = static_cast<float>(0.001);
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


cpp_fused_silu_46 = async_compile.cpp('''
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


cpp_fused_mul_sigmoid_silu_47 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_48 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (104L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_49 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_51 = async_compile.cpp('''
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


cpp_fused_convolution_52 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_53 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_54 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_55 = async_compile.cpp('''
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


cpp_fused_convolution_56 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_57 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_58 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_60 = async_compile.cpp('''
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


cpp_fused_convolution_61 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_62 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_63 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_64 = async_compile.cpp('''
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


cpp_fused_convolution_65 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_66 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_67 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_69 = async_compile.cpp('''
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


cpp_fused_convolution_70 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_71 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_72 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_73 = async_compile.cpp('''
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


cpp_fused_convolution_74 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_75 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_silu_76 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_mean_silu_77 = async_compile.cpp('''
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


cpp_fused_silu_78 = async_compile.cpp('''
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


cpp_fused_mul_sigmoid_silu_79 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_80 = async_compile.cpp('''
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_81 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_83 = async_compile.cpp('''
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


cpp_fused_convolution_84 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_85 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_86 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_87 = async_compile.cpp('''
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


cpp_fused_convolution_88 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_89 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_90 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_92 = async_compile.cpp('''
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


cpp_fused_convolution_93 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_94 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_95 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_96 = async_compile.cpp('''
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


cpp_fused_convolution_97 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_98 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_99 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused_convolution_101 = async_compile.cpp('''
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


cpp_fused_convolution_102 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_103 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_104 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_105 = async_compile.cpp('''
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


cpp_fused_convolution_106 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_107 = async_compile.cpp('''
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
                    auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_108 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
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
                    tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(15L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(15L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>(x1);
                            auto tmp1 = static_cast<int>(14);
                            auto tmp2 = tmp0 < tmp1;
                            auto tmp3 = c10::convert<int>(x2);
                            auto tmp4 = tmp3 < tmp1;
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = [&]
                            {
                                auto tmp7 = masked_load(in_out_ptr0 + static_cast<long>(x3 + (960L*x2) + (13440L*x1) + (188160L*x0)), to_float_mask(tmp5));
                                auto tmp8 = decltype(tmp7)(1)/(decltype(tmp7)(1) + tmp7.neg().exp());
                                auto tmp9 = tmp7 * tmp8;
                                return tmp9;
                            }
                            ;
                            auto tmp10 = decltype(tmp6())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp6(), to_float_mask(tmp5));
                            tmp10.store(out_ptr0 + static_cast<long>(x3 + (240L*x2) + (3600L*x1) + (54000L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_109 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(17L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(17L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-1L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(14);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-1L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-14160L) + x3 + (960L*x2) + (13440L*x1) + (188160L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (240L*x2) + (4080L*x1) + (69360L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_110 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(19L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(19L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-2L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(14);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-2L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-28320L) + x3 + (960L*x2) + (13440L*x1) + (188160L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (240L*x2) + (4560L*x1) + (86640L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused_constant_pad_nd_111 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(21L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(21L); x2+=static_cast<long>(1L))
                    {
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(240L); x3+=static_cast<long>(8L))
                        {
                            auto tmp0 = c10::convert<int>((-3L) + x1);
                            auto tmp1 = static_cast<int>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<int>(14);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = c10::convert<int>((-3L) + x2);
                            auto tmp6 = tmp5 >= tmp1;
                            auto tmp7 = tmp5 < tmp3;
                            auto tmp8 = tmp2 & tmp4;
                            auto tmp9 = tmp8 & tmp6;
                            auto tmp10 = tmp9 & tmp7;
                            auto tmp11 = [&]
                            {
                                auto tmp12 = masked_load(in_ptr0 + static_cast<long>((-42480L) + x3 + (960L*x2) + (13440L*x1) + (188160L*x0)), to_float_mask(tmp10));
                                auto tmp13 = decltype(tmp12)(1)/(decltype(tmp12)(1) + tmp12.neg().exp());
                                auto tmp14 = tmp12 * tmp13;
                                return tmp14;
                            }
                            ;
                            auto tmp15 = decltype(tmp11())::blendv(at::vec::Vectorized<float>(static_cast<float>(0.0)), tmp11(), to_float_mask(tmp10));
                            tmp15.store(out_ptr0 + static_cast<long>(x3 + (240L*x2) + (5040L*x1) + (105840L*x0)));
                        }
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_112 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_113 = async_compile.cpp('''
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


cpp_fused_mul_sigmoid_silu_114 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_115 = async_compile.cpp('''
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
                tmp16.store(in_out_ptr0 + static_cast<long>(x1 + (264L*x0)));
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_116 = async_compile.cpp('''
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


cpp_fused_convolution_118 = async_compile.cpp('''
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


cpp_fused_convolution_119 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_120 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_121 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_122 = async_compile.cpp('''
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


cpp_fused_convolution_123 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_124 = async_compile.cpp('''
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
                auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_125 = async_compile.cpp('''
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


cpp_fused_convolution_127 = async_compile.cpp('''
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


cpp_fused_convolution_128 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_129 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_130 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_131 = async_compile.cpp('''
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


cpp_fused_convolution_132 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_133 = async_compile.cpp('''
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
                auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_convolution_silu_134 = async_compile.cpp('''
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


cpp_fused_convolution_136 = async_compile.cpp('''
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


cpp_fused_convolution_137 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_138 = async_compile.cpp('''
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
                    auto tmp34 = static_cast<float>(0.001);
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


cpp_fused_silu_139 = async_compile.cpp('''
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


cpp_fused_convolution_mul_sigmoid_silu_140 = async_compile.cpp('''
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


cpp_fused_convolution_141 = async_compile.cpp('''
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


cpp_fused__native_batch_norm_legit_no_training_add_cat_142 = async_compile.cpp('''
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
                auto tmp18 = static_cast<float>(0.001);
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


cpp_fused__native_batch_norm_legit_no_training_mean_relu_143 = async_compile.cpp('''
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
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (32, ), (1, ))
    assert_size_stride(arg7_1, (192, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg10_1, (64, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg11_1, (64, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg12_1, (192, ), (1, ))
    assert_size_stride(arg13_1, (192, ), (1, ))
    assert_size_stride(arg14_1, (40, ), (1, ))
    assert_size_stride(arg15_1, (40, ), (1, ))
    assert_size_stride(arg16_1, (120, ), (1, ))
    assert_size_stride(arg17_1, (120, ), (1, ))
    assert_size_stride(arg18_1, (120, ), (1, ))
    assert_size_stride(arg19_1, (120, ), (1, ))
    assert_size_stride(arg20_1, (40, ), (1, ))
    assert_size_stride(arg21_1, (40, ), (1, ))
    assert_size_stride(arg22_1, (240, ), (1, ))
    assert_size_stride(arg23_1, (240, ), (1, ))
    assert_size_stride(arg24_1, (60, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg25_1, (60, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg26_1, (60, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg27_1, (60, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg28_1, (240, ), (1, ))
    assert_size_stride(arg29_1, (240, ), (1, ))
    assert_size_stride(arg30_1, (56, ), (1, ))
    assert_size_stride(arg31_1, (56, ), (1, ))
    assert_size_stride(arg32_1, (336, ), (1, ))
    assert_size_stride(arg33_1, (336, ), (1, ))
    assert_size_stride(arg34_1, (336, ), (1, ))
    assert_size_stride(arg35_1, (336, ), (1, ))
    assert_size_stride(arg36_1, (56, ), (1, ))
    assert_size_stride(arg37_1, (56, ), (1, ))
    assert_size_stride(arg38_1, (336, ), (1, ))
    assert_size_stride(arg39_1, (336, ), (1, ))
    assert_size_stride(arg40_1, (336, ), (1, ))
    assert_size_stride(arg41_1, (336, ), (1, ))
    assert_size_stride(arg42_1, (56, ), (1, ))
    assert_size_stride(arg43_1, (56, ), (1, ))
    assert_size_stride(arg44_1, (336, ), (1, ))
    assert_size_stride(arg45_1, (336, ), (1, ))
    assert_size_stride(arg46_1, (336, ), (1, ))
    assert_size_stride(arg47_1, (336, ), (1, ))
    assert_size_stride(arg48_1, (56, ), (1, ))
    assert_size_stride(arg49_1, (56, ), (1, ))
    assert_size_stride(arg50_1, (336, ), (1, ))
    assert_size_stride(arg51_1, (336, ), (1, ))
    assert_size_stride(arg52_1, (112, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg53_1, (112, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg54_1, (112, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg55_1, (336, ), (1, ))
    assert_size_stride(arg56_1, (336, ), (1, ))
    assert_size_stride(arg57_1, (104, ), (1, ))
    assert_size_stride(arg58_1, (104, ), (1, ))
    assert_size_stride(arg59_1, (624, ), (1, ))
    assert_size_stride(arg60_1, (624, ), (1, ))
    assert_size_stride(arg61_1, (624, ), (1, ))
    assert_size_stride(arg62_1, (624, ), (1, ))
    assert_size_stride(arg63_1, (104, ), (1, ))
    assert_size_stride(arg64_1, (104, ), (1, ))
    assert_size_stride(arg65_1, (624, ), (1, ))
    assert_size_stride(arg66_1, (624, ), (1, ))
    assert_size_stride(arg67_1, (624, ), (1, ))
    assert_size_stride(arg68_1, (624, ), (1, ))
    assert_size_stride(arg69_1, (104, ), (1, ))
    assert_size_stride(arg70_1, (104, ), (1, ))
    assert_size_stride(arg71_1, (624, ), (1, ))
    assert_size_stride(arg72_1, (624, ), (1, ))
    assert_size_stride(arg73_1, (624, ), (1, ))
    assert_size_stride(arg74_1, (624, ), (1, ))
    assert_size_stride(arg75_1, (104, ), (1, ))
    assert_size_stride(arg76_1, (104, ), (1, ))
    assert_size_stride(arg77_1, (624, ), (1, ))
    assert_size_stride(arg78_1, (624, ), (1, ))
    assert_size_stride(arg79_1, (624, ), (1, ))
    assert_size_stride(arg80_1, (624, ), (1, ))
    assert_size_stride(arg81_1, (160, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (480, ), (1, ))
    assert_size_stride(arg84_1, (480, ), (1, ))
    assert_size_stride(arg85_1, (480, ), (1, ))
    assert_size_stride(arg86_1, (480, ), (1, ))
    assert_size_stride(arg87_1, (160, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (480, ), (1, ))
    assert_size_stride(arg90_1, (480, ), (1, ))
    assert_size_stride(arg91_1, (480, ), (1, ))
    assert_size_stride(arg92_1, (480, ), (1, ))
    assert_size_stride(arg93_1, (160, ), (1, ))
    assert_size_stride(arg94_1, (160, ), (1, ))
    assert_size_stride(arg95_1, (480, ), (1, ))
    assert_size_stride(arg96_1, (480, ), (1, ))
    assert_size_stride(arg97_1, (480, ), (1, ))
    assert_size_stride(arg98_1, (480, ), (1, ))
    assert_size_stride(arg99_1, (160, ), (1, ))
    assert_size_stride(arg100_1, (160, ), (1, ))
    assert_size_stride(arg101_1, (960, ), (1, ))
    assert_size_stride(arg102_1, (960, ), (1, ))
    assert_size_stride(arg103_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg104_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (240, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg106_1, (240, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg107_1, (960, ), (1, ))
    assert_size_stride(arg108_1, (960, ), (1, ))
    assert_size_stride(arg109_1, (264, ), (1, ))
    assert_size_stride(arg110_1, (264, ), (1, ))
    assert_size_stride(arg111_1, (1584, ), (1, ))
    assert_size_stride(arg112_1, (1584, ), (1, ))
    assert_size_stride(arg113_1, (1584, ), (1, ))
    assert_size_stride(arg114_1, (1584, ), (1, ))
    assert_size_stride(arg115_1, (264, ), (1, ))
    assert_size_stride(arg116_1, (264, ), (1, ))
    assert_size_stride(arg117_1, (1584, ), (1, ))
    assert_size_stride(arg118_1, (1584, ), (1, ))
    assert_size_stride(arg119_1, (1584, ), (1, ))
    assert_size_stride(arg120_1, (1584, ), (1, ))
    assert_size_stride(arg121_1, (264, ), (1, ))
    assert_size_stride(arg122_1, (264, ), (1, ))
    assert_size_stride(arg123_1, (1584, ), (1, ))
    assert_size_stride(arg124_1, (1584, ), (1, ))
    assert_size_stride(arg125_1, (1584, ), (1, ))
    assert_size_stride(arg126_1, (1584, ), (1, ))
    assert_size_stride(arg127_1, (264, ), (1, ))
    assert_size_stride(arg128_1, (264, ), (1, ))
    assert_size_stride(arg129_1, (1536, ), (1, ))
    assert_size_stride(arg130_1, (1536, ), (1, ))
    assert_size_stride(arg131_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg132_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg133_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg134_1, (96, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg135_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg136_1, (20, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg137_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg138_1, (60, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg139_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg140_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg141_1, (20, 60, 1, 1), (60, 1, 1, 1))
    assert_size_stride(arg142_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg143_1, (20, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg144_1, (20, ), (1, ))
    assert_size_stride(arg145_1, (240, 20, 1, 1), (20, 1, 1, 1))
    assert_size_stride(arg146_1, (240, ), (1, ))
    assert_size_stride(arg147_1, (56, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg148_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg149_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg150_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg152_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg153_1, (28, ), (1, ))
    assert_size_stride(arg154_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg155_1, (336, ), (1, ))
    assert_size_stride(arg156_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg157_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg158_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg159_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg160_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg161_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg162_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg163_1, (28, ), (1, ))
    assert_size_stride(arg164_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg165_1, (336, ), (1, ))
    assert_size_stride(arg166_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg167_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg168_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg169_1, (168, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg170_1, (168, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg171_1, (168, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg172_1, (28, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg173_1, (28, ), (1, ))
    assert_size_stride(arg174_1, (336, 28, 1, 1), (28, 1, 1, 1))
    assert_size_stride(arg175_1, (336, ), (1, ))
    assert_size_stride(arg176_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg177_1, (28, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg178_1, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(arg179_1, (14, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg180_1, (14, ), (1, ))
    assert_size_stride(arg181_1, (336, 14, 1, 1), (14, 1, 1, 1))
    assert_size_stride(arg182_1, (336, ), (1, ))
    assert_size_stride(arg183_1, (104, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg184_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg185_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg186_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg187_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg188_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg189_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg190_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg191_1, (26, ), (1, ))
    assert_size_stride(arg192_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg193_1, (624, ), (1, ))
    assert_size_stride(arg194_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg195_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg196_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg197_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg198_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg199_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg200_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg201_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg202_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg203_1, (26, ), (1, ))
    assert_size_stride(arg204_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg205_1, (624, ), (1, ))
    assert_size_stride(arg206_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg207_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg208_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg209_1, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg210_1, (156, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg211_1, (156, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg212_1, (156, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg213_1, (156, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg214_1, (26, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg215_1, (26, ), (1, ))
    assert_size_stride(arg216_1, (624, 26, 1, 1), (26, 1, 1, 1))
    assert_size_stride(arg217_1, (624, ), (1, ))
    assert_size_stride(arg218_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg219_1, (52, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(arg220_1, (624, 104, 1, 1), (104, 1, 1, 1))
    assert_size_stride(arg221_1, (624, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg222_1, (52, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg223_1, (52, ), (1, ))
    assert_size_stride(arg224_1, (624, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(arg225_1, (624, ), (1, ))
    assert_size_stride(arg226_1, (160, 624, 1, 1), (624, 1, 1, 1))
    assert_size_stride(arg227_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg228_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg229_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg230_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg231_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg232_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg233_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg234_1, (80, ), (1, ))
    assert_size_stride(arg235_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg236_1, (480, ), (1, ))
    assert_size_stride(arg237_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg238_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg239_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg240_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg241_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg242_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg243_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg244_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg245_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg246_1, (80, ), (1, ))
    assert_size_stride(arg247_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg248_1, (480, ), (1, ))
    assert_size_stride(arg249_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg250_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg251_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg252_1, (240, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg253_1, (120, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg254_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg255_1, (120, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(arg256_1, (120, 1, 9, 9), (81, 81, 9, 1))
    assert_size_stride(arg257_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg258_1, (80, ), (1, ))
    assert_size_stride(arg259_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg260_1, (480, ), (1, ))
    assert_size_stride(arg261_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg262_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg263_1, (960, 160, 1, 1), (160, 1, 1, 1))
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
    buf0 = empty_strided((8, 3, 225, 225), (151875, 1, 675, 3), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_convolution_0(c_void_p(arg421_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    del arg421_1
    # Source Nodes: [x, x_1], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf2, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del buf0
    del buf1
    buf3 = buf2; del buf2  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_1(c_void_p(buf3.data_ptr()), c_void_p(arg305_1.data_ptr()), c_void_p(arg306_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg2_1.data_ptr()))
    del arg1_1
    del arg2_1
    del arg305_1
    del arg306_1
    # Source Nodes: [x_6], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf3, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
    assert_size_stride(buf4, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del arg131_1
    buf5 = buf4; del buf4  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_2(c_void_p(buf5.data_ptr()), c_void_p(arg307_1.data_ptr()), c_void_p(arg308_1.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(arg4_1.data_ptr()))
    del arg307_1
    del arg308_1
    del arg3_1
    del arg4_1
    # Source Nodes: [x_10, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
    buf6 = extern_kernels.convolution(buf5, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf6, (8, 32, 112, 112), (401408, 1, 3584, 32))
    del arg132_1
    buf7 = reinterpret_tensor(buf5, (8, 32, 112, 112), (401408, 12544, 112, 1), 0); del buf5  # reuse
    buf8 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_add_convolution_3(c_void_p(buf6.data_ptr()), c_void_p(arg309_1.data_ptr()), c_void_p(arg310_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg309_1
    del arg310_1
    del arg5_1
    del arg6_1
    del buf3
    del buf6
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_0], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf8, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg133_1
    buf10 = buf8; del buf8  # reuse
    cpp_fused_convolution_4(c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf7
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pw_1], Original ATen: [aten.convolution]
    buf11 = extern_kernels.convolution(buf10, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf11, (8, 96, 112, 112), (1204224, 1, 10752, 96))
    del arg134_1
    del buf10
    buf12 = empty_strided((8, 192, 112, 112), (2408448, 1, 21504, 192), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 64, 113, 113), (817216, 1, 7232, 64), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_constant_pad_nd_relu_5(c_void_p(buf9.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(arg311_1.data_ptr()), c_void_p(arg312_1.data_ptr()), c_void_p(arg7_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg311_1
    del arg312_1
    del arg7_1
    del arg8_1
    del buf11
    del buf9
    # Source Nodes: [conv2d_1, x_25], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf14 = extern_kernels.convolution(buf13, arg9_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf14, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg9_1
    del buf13
    buf15 = empty_strided((8, 64, 115, 115), (846400, 1, 7360, 64), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_6(c_void_p(buf12.data_ptr()), c_void_p(buf15.data_ptr()))
    # Source Nodes: [conv2d_2, x_27], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf16 = extern_kernels.convolution(buf15, arg10_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf16, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg10_1
    del buf15
    buf17 = empty_strided((8, 64, 117, 117), (876096, 1, 7488, 64), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_7(c_void_p(buf12.data_ptr()), c_void_p(buf17.data_ptr()))
    del buf12
    # Source Nodes: [conv2d_3, x_29], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf18 = extern_kernels.convolution(buf17, arg11_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
    assert_size_stride(buf18, (8, 64, 56, 56), (200704, 1, 3584, 64))
    del arg11_1
    del buf17
    buf19 = empty_strided((8, 192, 56, 56), (602112, 1, 10752, 192), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_8(c_void_p(buf14.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(arg313_1.data_ptr()), c_void_p(arg314_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(buf19.data_ptr()))
    del arg12_1
    del arg13_1
    del arg313_1
    del arg314_1
    del buf14
    del buf16
    del buf18
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_0], Original ATen: [aten.convolution]
    buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 96, 56, 56), (602112, 1, 10752, 192), 0), arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf20, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg135_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___conv_pwl_1], Original ATen: [aten.convolution]
    buf21 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 96, 56, 56), (602112, 1, 10752, 192), 96), arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf21, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg136_1
    del buf19
    buf22 = empty_strided((8, 40, 56, 56), (125440, 1, 2240, 40), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_9(c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()), c_void_p(arg315_1.data_ptr()), c_void_p(arg316_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf22.data_ptr()))
    del arg14_1
    del arg15_1
    del arg315_1
    del arg316_1
    del buf20
    del buf21
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf23 = extern_kernels.convolution(reinterpret_tensor(buf22, (8, 20, 56, 56), (125440, 1, 2240, 40), 0), arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf23, (8, 60, 56, 56), (188160, 1, 3360, 60))
    del arg137_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(reinterpret_tensor(buf22, (8, 20, 56, 56), (125440, 1, 2240, 40), 20), arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 60, 56, 56), (188160, 1, 3360, 60))
    del arg138_1
    buf25 = empty_strided((8, 120, 56, 56), (376320, 1, 6720, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_relu_10(c_void_p(buf23.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg317_1.data_ptr()), c_void_p(arg318_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()), c_void_p(buf25.data_ptr()))
    del arg16_1
    del arg17_1
    del arg317_1
    del arg318_1
    del buf23
    # Source Nodes: [cat_78, x_45, x_48, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
    buf26 = extern_kernels.convolution(buf25, arg139_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf26, (8, 120, 56, 56), (376320, 1, 6720, 120))
    del arg139_1
    del buf25
    buf27 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_relu_11(c_void_p(buf27.data_ptr()), c_void_p(arg319_1.data_ptr()), c_void_p(arg320_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg18_1
    del arg19_1
    del arg319_1
    del arg320_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf28 = extern_kernels.convolution(reinterpret_tensor(buf27, (8, 60, 56, 56), (376320, 1, 6720, 120), 0), arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf28, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg140_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(reinterpret_tensor(buf27, (8, 60, 56, 56), (376320, 1, 6720, 120), 60), arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 20, 56, 56), (62720, 1, 1120, 20))
    del arg141_1
    del buf27
    buf30 = buf22; del buf22  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_12(c_void_p(buf30.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg321_1.data_ptr()), c_void_p(arg322_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()))
    del arg20_1
    del arg21_1
    del arg321_1
    del arg322_1
    del buf28
    del buf29
    # Source Nodes: [cat_77, shortcut_3, x_57, x_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf31 = extern_kernels.convolution(buf30, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf31, (8, 240, 56, 56), (752640, 1, 13440, 240))
    del arg142_1
    del buf30
    buf32 = buf31; del buf31  # reuse
    buf33 = empty_strided((8, 60, 57, 57), (194940, 1, 3420, 60), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_13(c_void_p(buf32.data_ptr()), c_void_p(arg323_1.data_ptr()), c_void_p(arg324_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg22_1
    del arg23_1
    del arg323_1
    del arg324_1
    # Source Nodes: [conv2d_4, x_68], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf34 = extern_kernels.convolution(buf33, arg24_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf34, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg24_1
    del buf33
    buf35 = empty_strided((8, 60, 59, 59), (208860, 1, 3540, 60), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_14(c_void_p(buf32.data_ptr()), c_void_p(buf35.data_ptr()))
    # Source Nodes: [conv2d_5, x_70], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf36 = extern_kernels.convolution(buf35, arg25_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf36, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg25_1
    del buf35
    buf37 = empty_strided((8, 60, 61, 61), (223260, 1, 3660, 60), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_15(c_void_p(buf32.data_ptr()), c_void_p(buf37.data_ptr()))
    # Source Nodes: [conv2d_6, x_72], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf38 = extern_kernels.convolution(buf37, arg26_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf38, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg26_1
    del buf37
    buf39 = empty_strided((8, 60, 63, 63), (238140, 1, 3780, 60), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_16(c_void_p(buf32.data_ptr()), c_void_p(buf39.data_ptr()))
    del buf32
    # Source Nodes: [conv2d_7, x_74], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf40 = extern_kernels.convolution(buf39, arg27_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=60, bias=None)
    assert_size_stride(buf40, (8, 60, 28, 28), (47040, 1, 1680, 60))
    del arg27_1
    del buf39
    buf41 = reinterpret_tensor(buf24, (8, 240, 28, 28), (188160, 1, 6720, 240), 0); del buf24  # reuse
    buf42 = empty_strided((8, 240, 1, 1), (240, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf43 = reinterpret_tensor(buf42, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf42  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_17(c_void_p(buf43.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf38.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(arg325_1.data_ptr()), c_void_p(arg326_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(buf41.data_ptr()))
    del arg28_1
    del arg29_1
    del arg325_1
    del arg326_1
    del buf34
    del buf36
    del buf38
    del buf40
    # Source Nodes: [x_80, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf44 = extern_kernels.convolution(buf43, arg143_1, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf44, (8, 20, 1, 1), (20, 1, 20, 20))
    del arg143_1
    del arg144_1
    del buf43
    buf45 = buf44; del buf44  # reuse
    cpp_fused_silu_18(c_void_p(buf45.data_ptr()))
    # Source Nodes: [x_se_2, x_se_3], Original ATen: [aten.convolution, aten.silu]
    buf46 = extern_kernels.convolution(buf45, arg145_1, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf46, (8, 240, 1, 1), (240, 1, 240, 240))
    del arg145_1
    del arg146_1
    del buf45
    buf47 = buf41; del buf41  # reuse
    cpp_fused_mul_sigmoid_silu_19(c_void_p(buf47.data_ptr()), c_void_p(buf46.data_ptr()))
    del buf46
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_80, x_81, x_82], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf48 = extern_kernels.convolution(buf47, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf48, (8, 56, 28, 28), (43904, 1, 1568, 56))
    del arg147_1
    del buf47
    buf49 = buf48; del buf48  # reuse
    cpp_fused__native_batch_norm_legit_no_training_20(c_void_p(buf49.data_ptr()), c_void_p(arg327_1.data_ptr()), c_void_p(arg328_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg30_1
    del arg31_1
    del arg327_1
    del arg328_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf50 = extern_kernels.convolution(reinterpret_tensor(buf49, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf50, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg148_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf51 = extern_kernels.convolution(reinterpret_tensor(buf49, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf51, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg149_1
    buf52 = empty_strided((8, 336, 28, 28), (263424, 1, 9408, 336), device='cpu', dtype=torch.float32)
    buf53 = empty((8, 336, 28, 28), device='cpu', dtype=torch.float32)
    buf54 = empty_strided((8, 168, 28, 28), (131712, 1, 4704, 168), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_21(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(arg329_1.data_ptr()), c_void_p(arg330_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf54.data_ptr()))
    del arg329_1
    del arg32_1
    del arg330_1
    del arg33_1
    del buf50
    del buf51
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf55 = extern_kernels.convolution(buf54, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf55, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg150_1
    buf56 = buf54; del buf54  # reuse
    cpp_fused_convolution_22(c_void_p(buf53.data_ptr()), c_void_p(buf56.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, arg151_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf57, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg151_1
    del buf56
    buf58 = reinterpret_tensor(buf53, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf53  # reuse
    buf59 = empty_strided((8, 336, 1, 1), (336, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf60 = reinterpret_tensor(buf59, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf59  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_23(c_void_p(buf60.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf57.data_ptr()), c_void_p(arg331_1.data_ptr()), c_void_p(arg332_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(buf58.data_ptr()))
    del arg331_1
    del arg332_1
    del arg34_1
    del arg35_1
    del buf55
    # Source Nodes: [x_98, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf61 = extern_kernels.convolution(buf60, arg152_1, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf61, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg152_1
    del arg153_1
    del buf60
    buf62 = buf61; del buf61  # reuse
    cpp_fused_silu_24(c_void_p(buf62.data_ptr()))
    # Source Nodes: [x_se_6, x_se_7], Original ATen: [aten.convolution, aten.silu]
    buf63 = extern_kernels.convolution(buf62, arg154_1, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf63, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg154_1
    del arg155_1
    del buf62
    buf64 = reinterpret_tensor(buf52, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf52  # reuse
    buf65 = buf57; del buf57  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_25(c_void_p(buf58.data_ptr()), c_void_p(buf63.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(buf65.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf66 = extern_kernels.convolution(buf65, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf66, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg156_1
    buf67 = buf65; del buf65  # reuse
    cpp_fused_convolution_26(c_void_p(buf64.data_ptr()), c_void_p(buf67.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf68 = extern_kernels.convolution(buf67, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf68, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg157_1
    buf69 = buf49; del buf49  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_27(c_void_p(buf69.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(arg333_1.data_ptr()), c_void_p(arg334_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()))
    del arg333_1
    del arg334_1
    del arg36_1
    del arg37_1
    del buf66
    del buf68
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf70 = extern_kernels.convolution(reinterpret_tensor(buf69, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf70, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg158_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf71 = extern_kernels.convolution(reinterpret_tensor(buf69, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf71, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg159_1
    buf72 = reinterpret_tensor(buf64, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf64  # reuse
    buf73 = reinterpret_tensor(buf58, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf58  # reuse
    buf74 = buf67; del buf67  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_28(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg335_1.data_ptr()), c_void_p(arg336_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()), c_void_p(buf72.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf74.data_ptr()))
    del arg335_1
    del arg336_1
    del arg38_1
    del arg39_1
    del buf70
    del buf71
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf75 = extern_kernels.convolution(buf74, arg160_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf75, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg160_1
    buf76 = buf74; del buf74  # reuse
    cpp_fused_convolution_29(c_void_p(buf73.data_ptr()), c_void_p(buf76.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf77 = extern_kernels.convolution(buf76, arg161_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf77, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg161_1
    del buf76
    buf78 = reinterpret_tensor(buf73, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf73  # reuse
    buf79 = reinterpret_tensor(buf63, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf63  # reuse
    buf80 = reinterpret_tensor(buf79, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf79  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_30(c_void_p(buf80.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(arg337_1.data_ptr()), c_void_p(arg338_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg337_1
    del arg338_1
    del arg40_1
    del arg41_1
    del buf75
    # Source Nodes: [x_118, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf81 = extern_kernels.convolution(buf80, arg162_1, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf81, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg162_1
    del arg163_1
    del buf80
    buf82 = buf81; del buf81  # reuse
    cpp_fused_silu_31(c_void_p(buf82.data_ptr()))
    # Source Nodes: [x_se_10, x_se_11], Original ATen: [aten.convolution, aten.silu]
    buf83 = extern_kernels.convolution(buf82, arg164_1, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf83, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg164_1
    del arg165_1
    del buf82
    buf84 = reinterpret_tensor(buf72, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf72  # reuse
    buf85 = buf77; del buf77  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_32(c_void_p(buf78.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf86 = extern_kernels.convolution(buf85, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf86, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg166_1
    buf87 = buf85; del buf85  # reuse
    cpp_fused_convolution_33(c_void_p(buf84.data_ptr()), c_void_p(buf87.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf88 = extern_kernels.convolution(buf87, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf88, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg167_1
    buf89 = buf69; del buf69  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_34(c_void_p(buf89.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(buf88.data_ptr()), c_void_p(arg339_1.data_ptr()), c_void_p(arg340_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()))
    del arg339_1
    del arg340_1
    del arg42_1
    del arg43_1
    del buf86
    del buf88
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf90 = extern_kernels.convolution(reinterpret_tensor(buf89, (8, 28, 28, 28), (43904, 1, 1568, 56), 0), arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf90, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg168_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf91 = extern_kernels.convolution(reinterpret_tensor(buf89, (8, 28, 28, 28), (43904, 1, 1568, 56), 28), arg169_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf91, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg169_1
    buf92 = reinterpret_tensor(buf84, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf84  # reuse
    buf93 = reinterpret_tensor(buf78, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf78  # reuse
    buf94 = buf87; del buf87  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_35(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(arg341_1.data_ptr()), c_void_p(arg342_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del arg341_1
    del arg342_1
    del arg44_1
    del arg45_1
    del buf90
    del buf91
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf95 = extern_kernels.convolution(buf94, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf95, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg170_1
    buf96 = buf94; del buf94  # reuse
    cpp_fused_convolution_36(c_void_p(buf93.data_ptr()), c_void_p(buf96.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(buf96, arg171_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=168, bias=None)
    assert_size_stride(buf97, (8, 168, 28, 28), (131712, 1, 4704, 168))
    del arg171_1
    del buf96
    buf98 = reinterpret_tensor(buf93, (8, 336, 28, 28), (263424, 1, 9408, 336), 0); del buf93  # reuse
    buf99 = reinterpret_tensor(buf83, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf83  # reuse
    buf100 = reinterpret_tensor(buf99, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf99  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_37(c_void_p(buf100.data_ptr()), c_void_p(buf95.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(arg343_1.data_ptr()), c_void_p(arg344_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg343_1
    del arg344_1
    del arg46_1
    del arg47_1
    del buf95
    # Source Nodes: [x_138, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf101 = extern_kernels.convolution(buf100, arg172_1, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf101, (8, 28, 1, 1), (28, 1, 28, 28))
    del arg172_1
    del arg173_1
    del buf100
    buf102 = buf101; del buf101  # reuse
    cpp_fused_silu_38(c_void_p(buf102.data_ptr()))
    # Source Nodes: [x_se_14, x_se_15], Original ATen: [aten.convolution, aten.silu]
    buf103 = extern_kernels.convolution(buf102, arg174_1, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf103, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg174_1
    del arg175_1
    del buf102
    buf104 = reinterpret_tensor(buf92, (8, 336, 28, 28), (263424, 784, 28, 1), 0); del buf92  # reuse
    buf105 = buf97; del buf97  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_39(c_void_p(buf98.data_ptr()), c_void_p(buf103.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()))
    del buf98
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf106 = extern_kernels.convolution(buf105, arg176_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf106, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg176_1
    buf107 = buf105; del buf105  # reuse
    cpp_fused_convolution_40(c_void_p(buf104.data_ptr()), c_void_p(buf107.data_ptr()))
    del buf104
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf108 = extern_kernels.convolution(buf107, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf108, (8, 28, 28, 28), (21952, 1, 784, 28))
    del arg177_1
    del buf107
    buf109 = buf89; del buf89  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_41(c_void_p(buf109.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(arg345_1.data_ptr()), c_void_p(arg346_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()))
    del arg345_1
    del arg346_1
    del arg48_1
    del arg49_1
    del buf106
    del buf108
    # Source Nodes: [cat_67, shortcut_7, x_142, x_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf110 = extern_kernels.convolution(buf109, arg178_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf110, (8, 336, 28, 28), (263424, 1, 9408, 336))
    del arg178_1
    del buf109
    buf111 = buf110; del buf110  # reuse
    buf112 = empty_strided((8, 112, 29, 29), (94192, 1, 3248, 112), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_42(c_void_p(buf111.data_ptr()), c_void_p(arg347_1.data_ptr()), c_void_p(arg348_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(buf112.data_ptr()))
    del arg347_1
    del arg348_1
    del arg50_1
    del arg51_1
    # Source Nodes: [conv2d_8, x_153], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf113 = extern_kernels.convolution(buf112, arg52_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf113, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg52_1
    del buf112
    buf114 = empty_strided((8, 112, 31, 31), (107632, 1, 3472, 112), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_43(c_void_p(buf111.data_ptr()), c_void_p(buf114.data_ptr()))
    # Source Nodes: [conv2d_9, x_155], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf115 = extern_kernels.convolution(buf114, arg53_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf115, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg53_1
    del buf114
    buf116 = empty_strided((8, 112, 33, 33), (121968, 1, 3696, 112), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_44(c_void_p(buf111.data_ptr()), c_void_p(buf116.data_ptr()))
    del buf111
    # Source Nodes: [conv2d_10, x_157], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf117 = extern_kernels.convolution(buf116, arg54_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=112, bias=None)
    assert_size_stride(buf117, (8, 112, 14, 14), (21952, 1, 1568, 112))
    del arg54_1
    del buf116
    buf118 = empty_strided((8, 336, 14, 14), (65856, 1, 4704, 336), device='cpu', dtype=torch.float32)
    buf119 = reinterpret_tensor(buf103, (8, 336, 1, 1), (336, 1, 2688, 2688), 0); del buf103  # reuse
    buf120 = reinterpret_tensor(buf119, (8, 336, 1, 1), (336, 1, 336, 336), 0); del buf119  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_45(c_void_p(buf120.data_ptr()), c_void_p(buf113.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf117.data_ptr()), c_void_p(arg349_1.data_ptr()), c_void_p(arg350_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(buf118.data_ptr()))
    del arg349_1
    del arg350_1
    del arg55_1
    del arg56_1
    del buf113
    del buf115
    del buf117
    # Source Nodes: [x_163, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf121 = extern_kernels.convolution(buf120, arg179_1, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf121, (8, 14, 1, 1), (14, 1, 14, 14))
    del arg179_1
    del arg180_1
    del buf120
    buf122 = buf121; del buf121  # reuse
    cpp_fused_silu_46(c_void_p(buf122.data_ptr()))
    # Source Nodes: [x_se_18, x_se_19], Original ATen: [aten.convolution, aten.silu]
    buf123 = extern_kernels.convolution(buf122, arg181_1, arg182_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf123, (8, 336, 1, 1), (336, 1, 336, 336))
    del arg181_1
    del arg182_1
    del buf122
    buf124 = buf118; del buf118  # reuse
    cpp_fused_mul_sigmoid_silu_47(c_void_p(buf124.data_ptr()), c_void_p(buf123.data_ptr()))
    del buf123
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___se_gate, x_163, x_164, x_165], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf125 = extern_kernels.convolution(buf124, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf125, (8, 104, 14, 14), (20384, 1, 1456, 104))
    del arg183_1
    del buf124
    buf126 = buf125; del buf125  # reuse
    cpp_fused__native_batch_norm_legit_no_training_48(c_void_p(buf126.data_ptr()), c_void_p(arg351_1.data_ptr()), c_void_p(arg352_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg58_1.data_ptr()))
    del arg351_1
    del arg352_1
    del arg57_1
    del arg58_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf127 = extern_kernels.convolution(reinterpret_tensor(buf126, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg184_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf127, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg184_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf128 = extern_kernels.convolution(reinterpret_tensor(buf126, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf128, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg185_1
    buf129 = empty_strided((8, 624, 14, 14), (122304, 1, 8736, 624), device='cpu', dtype=torch.float32)
    buf130 = empty((8, 624, 14, 14), device='cpu', dtype=torch.float32)
    buf131 = empty_strided((8, 156, 14, 14), (30576, 1, 2184, 156), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_49(c_void_p(buf127.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(arg353_1.data_ptr()), c_void_p(arg354_1.data_ptr()), c_void_p(arg59_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(buf129.data_ptr()), c_void_p(buf130.data_ptr()), c_void_p(buf131.data_ptr()))
    del arg353_1
    del arg354_1
    del arg59_1
    del arg60_1
    del buf127
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf132 = extern_kernels.convolution(buf131, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf132, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg186_1
    buf133 = buf131; del buf131  # reuse
    cpp_fused_convolution_50(c_void_p(buf130.data_ptr()), c_void_p(buf133.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf134 = extern_kernels.convolution(buf133, arg187_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf134, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg187_1
    buf135 = buf133; del buf133  # reuse
    cpp_fused_convolution_51(c_void_p(buf130.data_ptr()), c_void_p(buf135.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf136 = extern_kernels.convolution(buf135, arg188_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf136, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg188_1
    buf137 = buf135; del buf135  # reuse
    cpp_fused_convolution_52(c_void_p(buf130.data_ptr()), c_void_p(buf137.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf138 = extern_kernels.convolution(buf137, arg189_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf138, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg189_1
    del buf137
    buf139 = reinterpret_tensor(buf130, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf130  # reuse
    buf140 = empty_strided((8, 624, 1, 1), (624, 1, 4992, 4992), device='cpu', dtype=torch.float32)
    buf141 = reinterpret_tensor(buf140, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf140  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_53(c_void_p(buf141.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(arg355_1.data_ptr()), c_void_p(arg356_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(buf139.data_ptr()))
    del arg355_1
    del arg356_1
    del arg61_1
    del arg62_1
    del buf132
    del buf134
    del buf136
    # Source Nodes: [x_181, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf142 = extern_kernels.convolution(buf141, arg190_1, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf142, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg190_1
    del arg191_1
    del buf141
    buf143 = buf142; del buf142  # reuse
    cpp_fused_silu_54(c_void_p(buf143.data_ptr()))
    # Source Nodes: [x_se_22, x_se_23], Original ATen: [aten.convolution, aten.silu]
    buf144 = extern_kernels.convolution(buf143, arg192_1, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf144, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg192_1
    del arg193_1
    del buf143
    buf145 = reinterpret_tensor(buf129, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf129  # reuse
    buf146 = buf128; del buf128  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_55(c_void_p(buf139.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(buf146.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf147 = extern_kernels.convolution(buf146, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf147, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg194_1
    buf148 = buf146; del buf146  # reuse
    cpp_fused_convolution_56(c_void_p(buf145.data_ptr()), c_void_p(buf148.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf149 = extern_kernels.convolution(buf148, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf149, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg195_1
    del buf148
    buf150 = buf126; del buf126  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_57(c_void_p(buf150.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf149.data_ptr()), c_void_p(arg357_1.data_ptr()), c_void_p(arg358_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg64_1.data_ptr()))
    del arg357_1
    del arg358_1
    del arg63_1
    del arg64_1
    del buf147
    del buf149
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf151 = extern_kernels.convolution(reinterpret_tensor(buf150, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf151, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg196_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf152 = extern_kernels.convolution(reinterpret_tensor(buf150, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf152, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg197_1
    buf153 = reinterpret_tensor(buf145, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf145  # reuse
    buf154 = reinterpret_tensor(buf139, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf139  # reuse
    buf155 = buf138; del buf138  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_58(c_void_p(buf151.data_ptr()), c_void_p(buf152.data_ptr()), c_void_p(arg359_1.data_ptr()), c_void_p(arg360_1.data_ptr()), c_void_p(arg65_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(buf153.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()))
    del arg359_1
    del arg360_1
    del arg65_1
    del arg66_1
    del buf151
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf156 = extern_kernels.convolution(buf155, arg198_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf156, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg198_1
    buf157 = buf155; del buf155  # reuse
    cpp_fused_convolution_59(c_void_p(buf154.data_ptr()), c_void_p(buf157.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf158 = extern_kernels.convolution(buf157, arg199_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf158, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg199_1
    buf159 = buf157; del buf157  # reuse
    cpp_fused_convolution_60(c_void_p(buf154.data_ptr()), c_void_p(buf159.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf160 = extern_kernels.convolution(buf159, arg200_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf160, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg200_1
    buf161 = buf159; del buf159  # reuse
    cpp_fused_convolution_61(c_void_p(buf154.data_ptr()), c_void_p(buf161.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf162 = extern_kernels.convolution(buf161, arg201_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf162, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg201_1
    del buf161
    buf163 = reinterpret_tensor(buf154, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf154  # reuse
    buf164 = reinterpret_tensor(buf144, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf144  # reuse
    buf165 = reinterpret_tensor(buf164, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf164  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_62(c_void_p(buf165.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf158.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(arg361_1.data_ptr()), c_void_p(arg362_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(buf163.data_ptr()))
    del arg361_1
    del arg362_1
    del arg67_1
    del arg68_1
    del buf156
    del buf158
    del buf160
    # Source Nodes: [x_201, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf166 = extern_kernels.convolution(buf165, arg202_1, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf166, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg202_1
    del arg203_1
    del buf165
    buf167 = buf166; del buf166  # reuse
    cpp_fused_silu_63(c_void_p(buf167.data_ptr()))
    # Source Nodes: [x_se_26, x_se_27], Original ATen: [aten.convolution, aten.silu]
    buf168 = extern_kernels.convolution(buf167, arg204_1, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf168, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg204_1
    del arg205_1
    del buf167
    buf169 = reinterpret_tensor(buf153, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf153  # reuse
    buf170 = buf152; del buf152  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_64(c_void_p(buf163.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf171 = extern_kernels.convolution(buf170, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf171, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg206_1
    buf172 = buf170; del buf170  # reuse
    cpp_fused_convolution_65(c_void_p(buf169.data_ptr()), c_void_p(buf172.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf173 = extern_kernels.convolution(buf172, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf173, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg207_1
    del buf172
    buf174 = buf150; del buf150  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_66(c_void_p(buf174.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(arg363_1.data_ptr()), c_void_p(arg364_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg70_1.data_ptr()))
    del arg363_1
    del arg364_1
    del arg69_1
    del arg70_1
    del buf171
    del buf173
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf175 = extern_kernels.convolution(reinterpret_tensor(buf174, (8, 52, 14, 14), (20384, 1, 1456, 104), 0), arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf175, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg208_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf176 = extern_kernels.convolution(reinterpret_tensor(buf174, (8, 52, 14, 14), (20384, 1, 1456, 104), 52), arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf176, (8, 312, 14, 14), (61152, 1, 4368, 312))
    del arg209_1
    buf177 = reinterpret_tensor(buf169, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf169  # reuse
    buf178 = reinterpret_tensor(buf163, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf163  # reuse
    buf179 = buf162; del buf162  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_67(c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(arg365_1.data_ptr()), c_void_p(arg366_1.data_ptr()), c_void_p(arg71_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(buf177.data_ptr()), c_void_p(buf178.data_ptr()), c_void_p(buf179.data_ptr()))
    del arg365_1
    del arg366_1
    del arg71_1
    del arg72_1
    del buf175
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf180 = extern_kernels.convolution(buf179, arg210_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf180, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg210_1
    buf181 = buf179; del buf179  # reuse
    cpp_fused_convolution_68(c_void_p(buf178.data_ptr()), c_void_p(buf181.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf182 = extern_kernels.convolution(buf181, arg211_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf182, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg211_1
    buf183 = buf181; del buf181  # reuse
    cpp_fused_convolution_69(c_void_p(buf178.data_ptr()), c_void_p(buf183.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf184 = extern_kernels.convolution(buf183, arg212_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf184, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg212_1
    buf185 = buf183; del buf183  # reuse
    cpp_fused_convolution_70(c_void_p(buf178.data_ptr()), c_void_p(buf185.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf186 = extern_kernels.convolution(buf185, arg213_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=156, bias=None)
    assert_size_stride(buf186, (8, 156, 14, 14), (30576, 1, 2184, 156))
    del arg213_1
    del buf185
    buf187 = reinterpret_tensor(buf178, (8, 624, 14, 14), (122304, 1, 8736, 624), 0); del buf178  # reuse
    buf188 = reinterpret_tensor(buf168, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf168  # reuse
    buf189 = reinterpret_tensor(buf188, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf188  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_71(c_void_p(buf189.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf182.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(arg367_1.data_ptr()), c_void_p(arg368_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(buf187.data_ptr()))
    del arg367_1
    del arg368_1
    del arg73_1
    del arg74_1
    del buf180
    del buf182
    del buf184
    del buf186
    # Source Nodes: [x_221, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf190 = extern_kernels.convolution(buf189, arg214_1, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf190, (8, 26, 1, 1), (26, 1, 26, 26))
    del arg214_1
    del arg215_1
    del buf189
    buf191 = buf190; del buf190  # reuse
    cpp_fused_silu_72(c_void_p(buf191.data_ptr()))
    # Source Nodes: [x_se_30, x_se_31], Original ATen: [aten.convolution, aten.silu]
    buf192 = extern_kernels.convolution(buf191, arg216_1, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf192, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg216_1
    del arg217_1
    del buf191
    buf193 = reinterpret_tensor(buf177, (8, 624, 14, 14), (122304, 196, 14, 1), 0); del buf177  # reuse
    buf194 = buf176; del buf176  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_73(c_void_p(buf187.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf194.data_ptr()))
    del buf187
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf195 = extern_kernels.convolution(buf194, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf195, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg218_1
    buf196 = buf194; del buf194  # reuse
    cpp_fused_convolution_74(c_void_p(buf193.data_ptr()), c_void_p(buf196.data_ptr()))
    del buf193
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf197 = extern_kernels.convolution(buf196, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf197, (8, 52, 14, 14), (10192, 1, 728, 52))
    del arg219_1
    del buf196
    buf198 = buf174; del buf174  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_75(c_void_p(buf198.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(arg369_1.data_ptr()), c_void_p(arg370_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg76_1.data_ptr()))
    del arg369_1
    del arg370_1
    del arg75_1
    del arg76_1
    del buf195
    del buf197
    # Source Nodes: [cat_57, shortcut_11, x_225, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf199 = extern_kernels.convolution(buf198, arg220_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf199, (8, 624, 14, 14), (122304, 1, 8736, 624))
    del arg220_1
    del buf198
    buf200 = buf199; del buf199  # reuse
    buf201 = buf200; del buf200  # reuse
    cpp_fused__native_batch_norm_legit_no_training_silu_76(c_void_p(buf201.data_ptr()), c_void_p(arg371_1.data_ptr()), c_void_p(arg372_1.data_ptr()), c_void_p(arg77_1.data_ptr()), c_void_p(arg78_1.data_ptr()))
    del arg371_1
    del arg372_1
    del arg77_1
    del arg78_1
    # Source Nodes: [x_234, x_235], Original ATen: [aten.convolution, aten.silu]
    buf202 = extern_kernels.convolution(buf201, arg221_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=624, bias=None)
    assert_size_stride(buf202, (8, 624, 14, 14), (122304, 1, 8736, 624))
    del arg221_1
    del buf201
    buf203 = buf202; del buf202  # reuse
    buf204 = reinterpret_tensor(buf192, (8, 624, 1, 1), (624, 1, 4992, 4992), 0); del buf192  # reuse
    buf205 = reinterpret_tensor(buf204, (8, 624, 1, 1), (624, 1, 624, 624), 0); del buf204  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_silu_77(c_void_p(buf203.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(arg373_1.data_ptr()), c_void_p(arg374_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(arg80_1.data_ptr()))
    del arg373_1
    del arg374_1
    del arg79_1
    del arg80_1
    # Source Nodes: [x_239, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf206 = extern_kernels.convolution(buf205, arg222_1, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf206, (8, 52, 1, 1), (52, 1, 52, 52))
    del arg222_1
    del arg223_1
    del buf205
    buf207 = buf206; del buf206  # reuse
    cpp_fused_silu_78(c_void_p(buf207.data_ptr()))
    # Source Nodes: [x_se_34, x_se_35], Original ATen: [aten.convolution, aten.silu]
    buf208 = extern_kernels.convolution(buf207, arg224_1, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf208, (8, 624, 1, 1), (624, 1, 624, 624))
    del arg224_1
    del arg225_1
    del buf207
    buf209 = buf203; del buf203  # reuse
    cpp_fused_mul_sigmoid_silu_79(c_void_p(buf209.data_ptr()), c_void_p(buf208.data_ptr()))
    del buf208
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_239, x_240, x_241], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf210 = extern_kernels.convolution(buf209, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf210, (8, 160, 14, 14), (31360, 1, 2240, 160))
    del arg226_1
    del buf209
    buf211 = buf210; del buf210  # reuse
    cpp_fused__native_batch_norm_legit_no_training_80(c_void_p(buf211.data_ptr()), c_void_p(arg375_1.data_ptr()), c_void_p(arg376_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg82_1.data_ptr()))
    del arg375_1
    del arg376_1
    del arg81_1
    del arg82_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_0], Original ATen: [aten.convolution]
    buf212 = extern_kernels.convolution(reinterpret_tensor(buf211, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf212, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg227_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pw_1], Original ATen: [aten.convolution]
    buf213 = extern_kernels.convolution(reinterpret_tensor(buf211, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf213, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg228_1
    buf214 = empty_strided((8, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    buf215 = empty((8, 480, 14, 14), device='cpu', dtype=torch.float32)
    buf216 = empty_strided((8, 120, 14, 14), (23520, 1, 1680, 120), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_81(c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()), c_void_p(arg377_1.data_ptr()), c_void_p(arg378_1.data_ptr()), c_void_p(arg83_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf215.data_ptr()), c_void_p(buf216.data_ptr()))
    del arg377_1
    del arg378_1
    del arg83_1
    del arg84_1
    del buf212
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf217 = extern_kernels.convolution(buf216, arg229_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf217, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg229_1
    buf218 = buf216; del buf216  # reuse
    cpp_fused_convolution_82(c_void_p(buf215.data_ptr()), c_void_p(buf218.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf219 = extern_kernels.convolution(buf218, arg230_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf219, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg230_1
    buf220 = buf218; del buf218  # reuse
    cpp_fused_convolution_83(c_void_p(buf215.data_ptr()), c_void_p(buf220.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf221 = extern_kernels.convolution(buf220, arg231_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf221, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg231_1
    buf222 = buf220; del buf220  # reuse
    cpp_fused_convolution_84(c_void_p(buf215.data_ptr()), c_void_p(buf222.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf223 = extern_kernels.convolution(buf222, arg232_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf223, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg232_1
    del buf222
    buf224 = reinterpret_tensor(buf215, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf215  # reuse
    buf225 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cpu', dtype=torch.float32)
    buf226 = reinterpret_tensor(buf225, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf225  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_85(c_void_p(buf226.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf223.data_ptr()), c_void_p(arg379_1.data_ptr()), c_void_p(arg380_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(buf224.data_ptr()))
    del arg379_1
    del arg380_1
    del arg85_1
    del arg86_1
    del buf217
    del buf219
    del buf221
    # Source Nodes: [x_257, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf227 = extern_kernels.convolution(buf226, arg233_1, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf227, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg233_1
    del arg234_1
    del buf226
    buf228 = buf227; del buf227  # reuse
    cpp_fused_silu_86(c_void_p(buf228.data_ptr()))
    # Source Nodes: [x_se_38, x_se_39], Original ATen: [aten.convolution, aten.silu]
    buf229 = extern_kernels.convolution(buf228, arg235_1, arg236_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf229, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg235_1
    del arg236_1
    del buf228
    buf230 = reinterpret_tensor(buf214, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf214  # reuse
    buf231 = buf213; del buf213  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_87(c_void_p(buf224.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(buf230.data_ptr()), c_void_p(buf231.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf232 = extern_kernels.convolution(buf231, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf232, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg237_1
    buf233 = buf231; del buf231  # reuse
    cpp_fused_convolution_88(c_void_p(buf230.data_ptr()), c_void_p(buf233.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf234 = extern_kernels.convolution(buf233, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf234, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg238_1
    del buf233
    buf235 = buf211; del buf211  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_89(c_void_p(buf235.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(arg381_1.data_ptr()), c_void_p(arg382_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg88_1.data_ptr()))
    del arg381_1
    del arg382_1
    del arg87_1
    del arg88_1
    del buf232
    del buf234
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_0], Original ATen: [aten.convolution]
    buf236 = extern_kernels.convolution(reinterpret_tensor(buf235, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf236, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg239_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pw_1], Original ATen: [aten.convolution]
    buf237 = extern_kernels.convolution(reinterpret_tensor(buf235, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf237, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg240_1
    buf238 = reinterpret_tensor(buf230, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf230  # reuse
    buf239 = reinterpret_tensor(buf224, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf224  # reuse
    buf240 = buf223; del buf223  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_90(c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(arg383_1.data_ptr()), c_void_p(arg384_1.data_ptr()), c_void_p(arg89_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(buf238.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()))
    del arg383_1
    del arg384_1
    del arg89_1
    del arg90_1
    del buf236
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf241 = extern_kernels.convolution(buf240, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf241, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg241_1
    buf242 = buf240; del buf240  # reuse
    cpp_fused_convolution_91(c_void_p(buf239.data_ptr()), c_void_p(buf242.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf243 = extern_kernels.convolution(buf242, arg242_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf243, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg242_1
    buf244 = buf242; del buf242  # reuse
    cpp_fused_convolution_92(c_void_p(buf239.data_ptr()), c_void_p(buf244.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf245 = extern_kernels.convolution(buf244, arg243_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf245, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg243_1
    buf246 = buf244; del buf244  # reuse
    cpp_fused_convolution_93(c_void_p(buf239.data_ptr()), c_void_p(buf246.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf247 = extern_kernels.convolution(buf246, arg244_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf247, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg244_1
    del buf246
    buf248 = reinterpret_tensor(buf239, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf239  # reuse
    buf249 = reinterpret_tensor(buf229, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf229  # reuse
    buf250 = reinterpret_tensor(buf249, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf249  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_94(c_void_p(buf250.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf245.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(arg385_1.data_ptr()), c_void_p(arg386_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(buf248.data_ptr()))
    del arg385_1
    del arg386_1
    del arg91_1
    del arg92_1
    del buf241
    del buf243
    del buf245
    # Source Nodes: [x_277, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf251 = extern_kernels.convolution(buf250, arg245_1, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf251, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg245_1
    del arg246_1
    del buf250
    buf252 = buf251; del buf251  # reuse
    cpp_fused_silu_95(c_void_p(buf252.data_ptr()))
    # Source Nodes: [x_se_42, x_se_43], Original ATen: [aten.convolution, aten.silu]
    buf253 = extern_kernels.convolution(buf252, arg247_1, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf253, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg247_1
    del arg248_1
    del buf252
    buf254 = reinterpret_tensor(buf238, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf238  # reuse
    buf255 = buf237; del buf237  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_96(c_void_p(buf248.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(buf255.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf256 = extern_kernels.convolution(buf255, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf256, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg249_1
    buf257 = buf255; del buf255  # reuse
    cpp_fused_convolution_97(c_void_p(buf254.data_ptr()), c_void_p(buf257.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf258 = extern_kernels.convolution(buf257, arg250_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf258, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg250_1
    del buf257
    buf259 = buf235; del buf235  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_98(c_void_p(buf259.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(arg387_1.data_ptr()), c_void_p(arg388_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg94_1.data_ptr()))
    del arg387_1
    del arg388_1
    del arg93_1
    del arg94_1
    del buf256
    del buf258
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_0], Original ATen: [aten.convolution]
    buf260 = extern_kernels.convolution(reinterpret_tensor(buf259, (8, 80, 14, 14), (31360, 1, 2240, 160), 0), arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf260, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg251_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pw_1], Original ATen: [aten.convolution]
    buf261 = extern_kernels.convolution(reinterpret_tensor(buf259, (8, 80, 14, 14), (31360, 1, 2240, 160), 80), arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf261, (8, 240, 14, 14), (47040, 1, 3360, 240))
    del arg252_1
    buf262 = reinterpret_tensor(buf254, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf254  # reuse
    buf263 = reinterpret_tensor(buf248, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf248  # reuse
    buf264 = buf247; del buf247  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_convolution_silu_99(c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(arg389_1.data_ptr()), c_void_p(arg390_1.data_ptr()), c_void_p(arg95_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(buf262.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()))
    del arg389_1
    del arg390_1
    del arg95_1
    del arg96_1
    del buf260
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf265 = extern_kernels.convolution(buf264, arg253_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf265, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg253_1
    buf266 = buf264; del buf264  # reuse
    cpp_fused_convolution_100(c_void_p(buf263.data_ptr()), c_void_p(buf266.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf267 = extern_kernels.convolution(buf266, arg254_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf267, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg254_1
    buf268 = buf266; del buf266  # reuse
    cpp_fused_convolution_101(c_void_p(buf263.data_ptr()), c_void_p(buf268.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf269 = extern_kernels.convolution(buf268, arg255_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf269, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg255_1
    buf270 = buf268; del buf268  # reuse
    cpp_fused_convolution_102(c_void_p(buf263.data_ptr()), c_void_p(buf270.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf271 = extern_kernels.convolution(buf270, arg256_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
    assert_size_stride(buf271, (8, 120, 14, 14), (23520, 1, 1680, 120))
    del arg256_1
    del buf270
    buf272 = reinterpret_tensor(buf263, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf263  # reuse
    buf273 = reinterpret_tensor(buf253, (8, 480, 1, 1), (480, 1, 3840, 3840), 0); del buf253  # reuse
    buf274 = reinterpret_tensor(buf273, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf273  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_103(c_void_p(buf274.data_ptr()), c_void_p(buf265.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf271.data_ptr()), c_void_p(arg391_1.data_ptr()), c_void_p(arg392_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(buf272.data_ptr()))
    del arg391_1
    del arg392_1
    del arg97_1
    del arg98_1
    del buf265
    del buf267
    del buf269
    del buf271
    # Source Nodes: [x_297, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf275 = extern_kernels.convolution(buf274, arg257_1, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf275, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg257_1
    del arg258_1
    del buf274
    buf276 = buf275; del buf275  # reuse
    cpp_fused_silu_104(c_void_p(buf276.data_ptr()))
    # Source Nodes: [x_se_46, x_se_47], Original ATen: [aten.convolution, aten.silu]
    buf277 = extern_kernels.convolution(buf276, arg259_1, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf277, (8, 480, 1, 1), (480, 1, 480, 480))
    del arg259_1
    del arg260_1
    del buf276
    buf278 = reinterpret_tensor(buf262, (8, 480, 14, 14), (94080, 196, 14, 1), 0); del buf262  # reuse
    buf279 = buf261; del buf261  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_105(c_void_p(buf272.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()))
    del buf272
    del buf277
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf280 = extern_kernels.convolution(buf279, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf280, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg261_1
    buf281 = buf279; del buf279  # reuse
    cpp_fused_convolution_106(c_void_p(buf278.data_ptr()), c_void_p(buf281.data_ptr()))
    del buf278
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf282 = extern_kernels.convolution(buf281, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf282, (8, 80, 14, 14), (15680, 1, 1120, 80))
    del arg262_1
    buf283 = buf259; del buf259  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_107(c_void_p(buf283.data_ptr()), c_void_p(buf280.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(arg393_1.data_ptr()), c_void_p(arg394_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg100_1.data_ptr()))
    del arg100_1
    del arg393_1
    del arg394_1
    del arg99_1
    del buf280
    del buf282
    # Source Nodes: [cat_48, shortcut_15, x_301, x_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf284 = extern_kernels.convolution(buf283, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf284, (8, 960, 14, 14), (188160, 1, 13440, 960))
    del arg263_1
    del buf283
    buf285 = buf284; del buf284  # reuse
    buf286 = empty_strided((8, 240, 15, 15), (54000, 1, 3600, 240), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_constant_pad_nd_108(c_void_p(buf285.data_ptr()), c_void_p(arg395_1.data_ptr()), c_void_p(arg396_1.data_ptr()), c_void_p(arg101_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(buf286.data_ptr()))
    del arg101_1
    del arg102_1
    del arg395_1
    del arg396_1
    # Source Nodes: [conv2d_11, x_312], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf287 = extern_kernels.convolution(buf286, arg103_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf287, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg103_1
    del buf286
    buf288 = empty_strided((8, 240, 17, 17), (69360, 1, 4080, 240), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_109(c_void_p(buf285.data_ptr()), c_void_p(buf288.data_ptr()))
    # Source Nodes: [conv2d_12, x_314], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf289 = extern_kernels.convolution(buf288, arg104_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf289, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg104_1
    del buf288
    buf290 = empty_strided((8, 240, 19, 19), (86640, 1, 4560, 240), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_110(c_void_p(buf285.data_ptr()), c_void_p(buf290.data_ptr()))
    # Source Nodes: [conv2d_13, x_316], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf291 = extern_kernels.convolution(buf290, arg105_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf291, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg105_1
    del buf290
    buf292 = empty_strided((8, 240, 21, 21), (105840, 1, 5040, 240), device='cpu', dtype=torch.float32)
    cpp_fused_constant_pad_nd_111(c_void_p(buf285.data_ptr()), c_void_p(buf292.data_ptr()))
    del buf285
    # Source Nodes: [conv2d_14, x_318], Original ATen: [aten.constant_pad_nd, aten.convolution]
    buf293 = extern_kernels.convolution(buf292, arg106_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
    assert_size_stride(buf293, (8, 240, 7, 7), (11760, 1, 1680, 240))
    del arg106_1
    del buf292
    buf294 = reinterpret_tensor(buf281, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf281  # reuse
    buf295 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cpu', dtype=torch.float32)
    buf296 = reinterpret_tensor(buf295, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf295  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_112(c_void_p(buf296.data_ptr()), c_void_p(buf287.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf293.data_ptr()), c_void_p(arg397_1.data_ptr()), c_void_p(arg398_1.data_ptr()), c_void_p(arg107_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(buf294.data_ptr()))
    del arg107_1
    del arg108_1
    del arg397_1
    del arg398_1
    del buf287
    del buf289
    del buf291
    del buf293
    # Source Nodes: [x_324, x_se_48, x_se_49], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf297 = extern_kernels.convolution(buf296, arg264_1, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf297, (8, 80, 1, 1), (80, 1, 80, 80))
    del arg264_1
    del arg265_1
    del buf296
    buf298 = buf297; del buf297  # reuse
    cpp_fused_silu_113(c_void_p(buf298.data_ptr()))
    # Source Nodes: [x_se_50, x_se_51], Original ATen: [aten.convolution, aten.silu]
    buf299 = extern_kernels.convolution(buf298, arg266_1, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf299, (8, 960, 1, 1), (960, 1, 960, 960))
    del arg266_1
    del arg267_1
    del buf298
    buf300 = buf294; del buf294  # reuse
    cpp_fused_mul_sigmoid_silu_114(c_void_p(buf300.data_ptr()), c_void_p(buf299.data_ptr()))
    del buf299
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_324, x_325, x_326], Original ATen: [aten.convolution, aten.mul, aten.sigmoid, aten.silu]
    buf301 = extern_kernels.convolution(buf300, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf301, (8, 264, 7, 7), (12936, 1, 1848, 264))
    del arg268_1
    del buf300
    buf302 = buf301; del buf301  # reuse
    cpp_fused__native_batch_norm_legit_no_training_115(c_void_p(buf302.data_ptr()), c_void_p(arg399_1.data_ptr()), c_void_p(arg400_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(arg110_1.data_ptr()))
    del arg109_1
    del arg110_1
    del arg399_1
    del arg400_1
    # Source Nodes: [x_331], Original ATen: [aten.convolution]
    buf303 = extern_kernels.convolution(buf302, arg269_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf303, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg269_1
    buf304 = buf303; del buf303  # reuse
    buf305 = empty((8, 1584, 7, 7), device='cpu', dtype=torch.float32)
    buf306 = empty_strided((8, 396, 7, 7), (19404, 1, 2772, 396), device='cpu', dtype=torch.float32)
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_116(c_void_p(buf304.data_ptr()), c_void_p(arg401_1.data_ptr()), c_void_p(arg402_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(buf306.data_ptr()))
    del arg111_1
    del arg112_1
    del arg401_1
    del arg402_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_0], Original ATen: [aten.convolution]
    buf307 = extern_kernels.convolution(buf306, arg270_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf307, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg270_1
    buf308 = buf306; del buf306  # reuse
    cpp_fused_convolution_117(c_void_p(buf305.data_ptr()), c_void_p(buf308.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_1], Original ATen: [aten.convolution]
    buf309 = extern_kernels.convolution(buf308, arg271_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf309, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg271_1
    buf310 = buf308; del buf308  # reuse
    cpp_fused_convolution_118(c_void_p(buf305.data_ptr()), c_void_p(buf310.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_2], Original ATen: [aten.convolution]
    buf311 = extern_kernels.convolution(buf310, arg272_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf311, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg272_1
    buf312 = buf310; del buf310  # reuse
    cpp_fused_convolution_119(c_void_p(buf305.data_ptr()), c_void_p(buf312.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_dw_3], Original ATen: [aten.convolution]
    buf313 = extern_kernels.convolution(buf312, arg273_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf313, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg273_1
    del buf312
    buf314 = reinterpret_tensor(buf305, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf305  # reuse
    buf315 = empty_strided((8, 1584, 1, 1), (1584, 1, 12672, 12672), device='cpu', dtype=torch.float32)
    buf316 = reinterpret_tensor(buf315, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf315  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_120(c_void_p(buf316.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf309.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(arg403_1.data_ptr()), c_void_p(arg404_1.data_ptr()), c_void_p(arg113_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(buf314.data_ptr()))
    del arg113_1
    del arg114_1
    del arg403_1
    del arg404_1
    del buf307
    del buf309
    del buf311
    # Source Nodes: [x_341, x_se_52, x_se_53], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf317 = extern_kernels.convolution(buf316, arg274_1, arg275_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf317, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg274_1
    del arg275_1
    del buf316
    buf318 = buf317; del buf317  # reuse
    cpp_fused_silu_121(c_void_p(buf318.data_ptr()))
    # Source Nodes: [x_se_54, x_se_55], Original ATen: [aten.convolution, aten.silu]
    buf319 = extern_kernels.convolution(buf318, arg276_1, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf319, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg276_1
    del arg277_1
    del buf318
    buf320 = reinterpret_tensor(buf304, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf304  # reuse
    buf321 = empty_strided((8, 792, 7, 7), (38808, 1, 5544, 792), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_mul_sigmoid_silu_122(c_void_p(buf314.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()))
    del buf314
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_0], Original ATen: [aten.convolution]
    buf322 = extern_kernels.convolution(buf321, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf322, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg278_1
    buf323 = buf321; del buf321  # reuse
    cpp_fused_convolution_123(c_void_p(buf320.data_ptr()), c_void_p(buf323.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___conv_pwl_1], Original ATen: [aten.convolution]
    buf324 = extern_kernels.convolution(buf323, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf324, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg279_1
    buf325 = buf302; del buf302  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_124(c_void_p(buf325.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(arg405_1.data_ptr()), c_void_p(arg406_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(arg116_1.data_ptr()))
    del arg115_1
    del arg116_1
    del arg405_1
    del arg406_1
    del buf322
    del buf324
    # Source Nodes: [x_350], Original ATen: [aten.convolution]
    buf326 = extern_kernels.convolution(buf325, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf326, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg280_1
    buf327 = buf326; del buf326  # reuse
    buf328 = buf320; del buf320  # reuse
    buf329 = buf313; del buf313  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_125(c_void_p(buf327.data_ptr()), c_void_p(arg407_1.data_ptr()), c_void_p(arg408_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()))
    del arg117_1
    del arg118_1
    del arg407_1
    del arg408_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_0], Original ATen: [aten.convolution]
    buf330 = extern_kernels.convolution(buf329, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf330, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg281_1
    buf331 = buf329; del buf329  # reuse
    cpp_fused_convolution_126(c_void_p(buf328.data_ptr()), c_void_p(buf331.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_1], Original ATen: [aten.convolution]
    buf332 = extern_kernels.convolution(buf331, arg282_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf332, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg282_1
    buf333 = buf331; del buf331  # reuse
    cpp_fused_convolution_127(c_void_p(buf328.data_ptr()), c_void_p(buf333.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_2], Original ATen: [aten.convolution]
    buf334 = extern_kernels.convolution(buf333, arg283_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf334, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg283_1
    buf335 = buf333; del buf333  # reuse
    cpp_fused_convolution_128(c_void_p(buf328.data_ptr()), c_void_p(buf335.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_dw_3], Original ATen: [aten.convolution]
    buf336 = extern_kernels.convolution(buf335, arg284_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf336, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg284_1
    del buf335
    buf337 = reinterpret_tensor(buf328, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf328  # reuse
    buf338 = reinterpret_tensor(buf319, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf319  # reuse
    buf339 = reinterpret_tensor(buf338, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf338  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_129(c_void_p(buf339.data_ptr()), c_void_p(buf330.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(arg409_1.data_ptr()), c_void_p(arg410_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(buf337.data_ptr()))
    del arg119_1
    del arg120_1
    del arg409_1
    del arg410_1
    del buf330
    del buf332
    del buf334
    # Source Nodes: [x_360, x_se_56, x_se_57], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf340 = extern_kernels.convolution(buf339, arg285_1, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf340, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg285_1
    del arg286_1
    del buf339
    buf341 = buf340; del buf340  # reuse
    cpp_fused_silu_130(c_void_p(buf341.data_ptr()))
    # Source Nodes: [x_se_58, x_se_59], Original ATen: [aten.convolution, aten.silu]
    buf342 = extern_kernels.convolution(buf341, arg287_1, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf342, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg287_1
    del arg288_1
    del buf341
    buf343 = reinterpret_tensor(buf327, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf327  # reuse
    buf344 = buf323; del buf323  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_131(c_void_p(buf337.data_ptr()), c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf344.data_ptr()))
    del buf337
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_0], Original ATen: [aten.convolution]
    buf345 = extern_kernels.convolution(buf344, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf345, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg289_1
    buf346 = buf344; del buf344  # reuse
    cpp_fused_convolution_132(c_void_p(buf343.data_ptr()), c_void_p(buf346.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___conv_pwl_1], Original ATen: [aten.convolution]
    buf347 = extern_kernels.convolution(buf346, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf347, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg290_1
    buf348 = buf325; del buf325  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_133(c_void_p(buf348.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(arg411_1.data_ptr()), c_void_p(arg412_1.data_ptr()), c_void_p(arg121_1.data_ptr()), c_void_p(arg122_1.data_ptr()))
    del arg121_1
    del arg122_1
    del arg411_1
    del arg412_1
    del buf345
    del buf347
    # Source Nodes: [x_369], Original ATen: [aten.convolution]
    buf349 = extern_kernels.convolution(buf348, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf349, (8, 1584, 7, 7), (77616, 1, 11088, 1584))
    del arg291_1
    buf350 = buf349; del buf349  # reuse
    buf351 = buf343; del buf343  # reuse
    buf352 = buf336; del buf336  # reuse
    cpp_fused__native_batch_norm_legit_no_training_convolution_silu_134(c_void_p(buf350.data_ptr()), c_void_p(arg413_1.data_ptr()), c_void_p(arg414_1.data_ptr()), c_void_p(arg123_1.data_ptr()), c_void_p(arg124_1.data_ptr()), c_void_p(buf351.data_ptr()), c_void_p(buf352.data_ptr()))
    del arg123_1
    del arg124_1
    del arg413_1
    del arg414_1
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_0], Original ATen: [aten.convolution]
    buf353 = extern_kernels.convolution(buf352, arg292_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf353, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg292_1
    buf354 = buf352; del buf352  # reuse
    cpp_fused_convolution_135(c_void_p(buf351.data_ptr()), c_void_p(buf354.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_1], Original ATen: [aten.convolution]
    buf355 = extern_kernels.convolution(buf354, arg293_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf355, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg293_1
    buf356 = buf354; del buf354  # reuse
    cpp_fused_convolution_136(c_void_p(buf351.data_ptr()), c_void_p(buf356.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_2], Original ATen: [aten.convolution]
    buf357 = extern_kernels.convolution(buf356, arg294_1, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf357, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg294_1
    buf358 = buf356; del buf356  # reuse
    cpp_fused_convolution_137(c_void_p(buf351.data_ptr()), c_void_p(buf358.data_ptr()))
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_dw_3], Original ATen: [aten.convolution]
    buf359 = extern_kernels.convolution(buf358, arg295_1, stride=(1, 1), padding=(4, 4), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=396, bias=None)
    assert_size_stride(buf359, (8, 396, 7, 7), (19404, 1, 2772, 396))
    del arg295_1
    del buf358
    buf360 = reinterpret_tensor(buf351, (8, 1584, 7, 7), (77616, 1, 11088, 1584), 0); del buf351  # reuse
    buf361 = reinterpret_tensor(buf342, (8, 1584, 1, 1), (1584, 1, 12672, 12672), 0); del buf342  # reuse
    buf362 = reinterpret_tensor(buf361, (8, 1584, 1, 1), (1584, 1, 1584, 1584), 0); del buf361  # reuse
    cpp_fused__native_batch_norm_legit_no_training_cat_mean_silu_138(c_void_p(buf362.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(arg415_1.data_ptr()), c_void_p(arg416_1.data_ptr()), c_void_p(arg125_1.data_ptr()), c_void_p(arg126_1.data_ptr()), c_void_p(buf360.data_ptr()))
    del arg125_1
    del arg126_1
    del arg415_1
    del arg416_1
    del buf353
    del buf355
    del buf357
    del buf359
    # Source Nodes: [x_379, x_se_60, x_se_61], Original ATen: [aten.convolution, aten.mean, aten.silu]
    buf363 = extern_kernels.convolution(buf362, arg296_1, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf363, (8, 132, 1, 1), (132, 1, 132, 132))
    del arg296_1
    del arg297_1
    del buf362
    buf364 = buf363; del buf363  # reuse
    cpp_fused_silu_139(c_void_p(buf364.data_ptr()))
    # Source Nodes: [x_se_62, x_se_63], Original ATen: [aten.convolution, aten.silu]
    buf365 = extern_kernels.convolution(buf364, arg298_1, arg299_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1)
    assert_size_stride(buf365, (8, 1584, 1, 1), (1584, 1, 1584, 1584))
    del arg298_1
    del arg299_1
    del buf364
    buf366 = reinterpret_tensor(buf350, (8, 1584, 7, 7), (77616, 49, 7, 1), 0); del buf350  # reuse
    buf367 = buf346; del buf346  # reuse
    cpp_fused_convolution_mul_sigmoid_silu_140(c_void_p(buf360.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()))
    del buf360
    del buf365
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_0], Original ATen: [aten.convolution]
    buf368 = extern_kernels.convolution(buf367, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf368, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg300_1
    buf369 = buf367; del buf367  # reuse
    cpp_fused_convolution_141(c_void_p(buf366.data_ptr()), c_void_p(buf369.data_ptr()))
    del buf366
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____3___conv_pwl_1], Original ATen: [aten.convolution]
    buf370 = extern_kernels.convolution(buf369, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf370, (8, 132, 7, 7), (6468, 1, 924, 132))
    del arg301_1
    del buf369
    buf371 = buf348; del buf348  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_cat_142(c_void_p(buf371.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(buf370.data_ptr()), c_void_p(arg417_1.data_ptr()), c_void_p(arg418_1.data_ptr()), c_void_p(arg127_1.data_ptr()), c_void_p(arg128_1.data_ptr()))
    del arg127_1
    del arg128_1
    del arg417_1
    del arg418_1
    del buf368
    del buf370
    # Source Nodes: [cat_41, x_383, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.cat, aten.convolution]
    buf372 = extern_kernels.convolution(buf371, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf372, (8, 1536, 7, 7), (75264, 1, 10752, 1536))
    del arg302_1
    del buf371
    buf373 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cpu', dtype=torch.float32)
    buf374 = reinterpret_tensor(buf373, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf373  # reuse
    cpp_fused__native_batch_norm_legit_no_training_mean_relu_143(c_void_p(buf374.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(arg419_1.data_ptr()), c_void_p(arg420_1.data_ptr()), c_void_p(arg129_1.data_ptr()), c_void_p(arg130_1.data_ptr()))
    del arg129_1
    del arg130_1
    del arg419_1
    del arg420_1
    del buf372
    buf375 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_398], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg304_1, reinterpret_tensor(buf374, (8, 1536), (1536, 1), 0), reinterpret_tensor(arg303_1, (1536, 1000), (1, 1536), 0), alpha=1, beta=1, out=buf375)
    del arg303_1
    del arg304_1
    return (buf375, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((64, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((64, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((60, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((60, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((60, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((60, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((56, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((112, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((112, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((112, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((104, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((240, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((240, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((1584, ), (1, ), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((264, ), (1, ), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((96, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((20, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((60, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((20, 60, 1, 1), (60, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((20, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((20, ), (1, ), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((240, 20, 1, 1), (20, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((56, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((168, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((168, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((168, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((28, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((28, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((336, 28, 1, 1), (28, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((28, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((14, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((14, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((336, 14, 1, 1), (14, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((336, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((104, 336, 1, 1), (336, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((156, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((156, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((156, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((156, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((26, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((26, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((624, 26, 1, 1), (26, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((52, 312, 1, 1), (312, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((624, 104, 1, 1), (104, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((624, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((52, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((52, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((624, 52, 1, 1), (52, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((624, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((160, 624, 1, 1), (624, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((240, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((120, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((120, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((120, 1, 9, 9), (81, 81, 9, 1), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
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
    compiled_module_main('tf_mixnet_l', benchmark_compiled_module)
