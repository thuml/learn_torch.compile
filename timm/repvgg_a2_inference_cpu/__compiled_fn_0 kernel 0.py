
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
                        out_ptr1[static_cast<long>(x1 + (3L*x2) + (150528L*x0))] = tmp0;
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                    auto tmp0 = in_ptr0[static_cast<long>(x2 + (9L*x1) + (27L*x0))];
                    out_ptr0[static_cast<long>(x1 + (3L*x2) + (27L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(100352L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (64L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6422528L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)), static_cast<long>(64L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (576L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (64L*x2) + (576L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (96L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2408448L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                        tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                    }
                    at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                }
                #pragma GCC ivdep
                for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(25088L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(96L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (96L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(1L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)), static_cast<long>(96L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (864L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (96L*x2) + (864L*x0)));
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
                       const float* in_ptr8)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1204224L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
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
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(6272L); x0+=static_cast<long>(1L))
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
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_15 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)), static_cast<long>(192L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (1728L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (192L*x2) + (1728L*x0)));
                    }
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(602112L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                tmp1.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}
''')


cpp_fused_convolution_17 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_19 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
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
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_21 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_23 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_25 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_35 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_37 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_39 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
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
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       const float* in_ptr8,
                       const float* in_ptr9,
                       const float* in_ptr10,
                       const float* in_ptr11,
                       const float* in_ptr12,
                       const float* in_ptr13)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr13 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(384L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_relu_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
                       const float* in_ptr12)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1568L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp28 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                    auto tmp32 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp34 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp40 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp42 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x1));
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
                    auto tmp33 = tmp31 - tmp32;
                    auto tmp35 = tmp34 + tmp5;
                    auto tmp36 = tmp35.sqrt();
                    auto tmp37 = tmp36.reciprocal();
                    auto tmp38 = tmp37 * tmp10;
                    auto tmp39 = tmp33 * tmp38;
                    auto tmp41 = tmp39 * tmp40;
                    auto tmp43 = tmp41 + tmp42;
                    auto tmp44 = tmp30 + tmp43;
                    auto tmp45 = at::vec::clamp_min(tmp44, decltype(tmp44)(0));
                    tmp45.store(in_out_ptr1 + static_cast<long>(x1 + (384L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1408L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(384L); x1+=static_cast<long>(8L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(8L))
                    {
                        float tmp1[8*8] __attribute__ ((aligned (8)));
                        for (long x1_inner = 0; x1_inner < 8; x1_inner++)
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0)));
                            tmp0.store(tmp1 + static_cast<long>(8L*x1_inner));
                        }
                        at::vec::transpose_mxn<float,8,8>(tmp1, 8, out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)), static_cast<long>(384L));
                    }
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(8L); x2<static_cast<long>(9L); x2+=static_cast<long>(1L))
                    {
                        auto tmp0 = ([&]() { __at_align__ float tmpbuf[8]; for (long x1_inner = 0; x1_inner < 8; x1_inner++) tmpbuf[x1_inner] = in_ptr0[static_cast<long>(x2 + (9L*x1) + (9L*x1_inner) + (3456L*x0))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                        tmp0.store(out_ptr0 + static_cast<long>(x1 + (384L*x2) + (3456L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
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
    auto out_ptr0 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(392L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1408L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (1408L*x0)));
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
                    tmp30.store(in_out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1408L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1408L*x2) + (68992L*x0)));
                            auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                            tmp_acc0_vec = tmp_acc0_vec + tmp1;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (1408L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(11264L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = static_cast<float>(49.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = tmp0 / tmp2;
                    tmp3.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, ), (1, ))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (96, ), (1, ))
    assert_size_stride(arg5_1, (96, ), (1, ))
    assert_size_stride(arg6_1, (96, ), (1, ))
    assert_size_stride(arg7_1, (96, ), (1, ))
    assert_size_stride(arg8_1, (96, ), (1, ))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (96, ), (1, ))
    assert_size_stride(arg12_1, (96, ), (1, ))
    assert_size_stride(arg13_1, (96, ), (1, ))
    assert_size_stride(arg14_1, (192, ), (1, ))
    assert_size_stride(arg15_1, (192, ), (1, ))
    assert_size_stride(arg16_1, (192, ), (1, ))
    assert_size_stride(arg17_1, (192, ), (1, ))
    assert_size_stride(arg18_1, (192, ), (1, ))
    assert_size_stride(arg19_1, (192, ), (1, ))
    assert_size_stride(arg20_1, (192, ), (1, ))
    assert_size_stride(arg21_1, (192, ), (1, ))
    assert_size_stride(arg22_1, (192, ), (1, ))
    assert_size_stride(arg23_1, (192, ), (1, ))
    assert_size_stride(arg24_1, (192, ), (1, ))
    assert_size_stride(arg25_1, (192, ), (1, ))
    assert_size_stride(arg26_1, (192, ), (1, ))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (384, ), (1, ))
    assert_size_stride(arg37_1, (384, ), (1, ))
    assert_size_stride(arg38_1, (384, ), (1, ))
    assert_size_stride(arg39_1, (384, ), (1, ))
    assert_size_stride(arg40_1, (384, ), (1, ))
    assert_size_stride(arg41_1, (384, ), (1, ))
    assert_size_stride(arg42_1, (384, ), (1, ))
    assert_size_stride(arg43_1, (384, ), (1, ))
    assert_size_stride(arg44_1, (384, ), (1, ))
    assert_size_stride(arg45_1, (384, ), (1, ))
    assert_size_stride(arg46_1, (384, ), (1, ))
    assert_size_stride(arg47_1, (384, ), (1, ))
    assert_size_stride(arg48_1, (384, ), (1, ))
    assert_size_stride(arg49_1, (384, ), (1, ))
    assert_size_stride(arg50_1, (384, ), (1, ))
    assert_size_stride(arg51_1, (384, ), (1, ))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (384, ), (1, ))
    assert_size_stride(arg55_1, (384, ), (1, ))
    assert_size_stride(arg56_1, (384, ), (1, ))
    assert_size_stride(arg57_1, (384, ), (1, ))
    assert_size_stride(arg58_1, (384, ), (1, ))
    assert_size_stride(arg59_1, (384, ), (1, ))
    assert_size_stride(arg60_1, (384, ), (1, ))
    assert_size_stride(arg61_1, (384, ), (1, ))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, ), (1, ))
    assert_size_stride(arg64_1, (384, ), (1, ))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (384, ), (1, ))
    assert_size_stride(arg67_1, (384, ), (1, ))
    assert_size_stride(arg68_1, (384, ), (1, ))
    assert_size_stride(arg69_1, (384, ), (1, ))
    assert_size_stride(arg70_1, (384, ), (1, ))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, ), (1, ))
    assert_size_stride(arg73_1, (384, ), (1, ))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, ), (1, ))
    assert_size_stride(arg76_1, (384, ), (1, ))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (384, ), (1, ))
    assert_size_stride(arg79_1, (384, ), (1, ))
    assert_size_stride(arg80_1, (384, ), (1, ))
    assert_size_stride(arg81_1, (384, ), (1, ))
    assert_size_stride(arg82_1, (384, ), (1, ))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, ), (1, ))
    assert_size_stride(arg85_1, (384, ), (1, ))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, ), (1, ))
    assert_size_stride(arg88_1, (384, ), (1, ))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (384, ), (1, ))
    assert_size_stride(arg93_1, (384, ), (1, ))
    assert_size_stride(arg94_1, (384, ), (1, ))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, ), (1, ))
    assert_size_stride(arg97_1, (384, ), (1, ))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, ), (1, ))
    assert_size_stride(arg100_1, (384, ), (1, ))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (384, ), (1, ))
    assert_size_stride(arg103_1, (384, ), (1, ))
    assert_size_stride(arg104_1, (384, ), (1, ))
    assert_size_stride(arg105_1, (384, ), (1, ))
    assert_size_stride(arg106_1, (384, ), (1, ))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, ), (1, ))
    assert_size_stride(arg109_1, (384, ), (1, ))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, ), (1, ))
    assert_size_stride(arg112_1, (384, ), (1, ))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (384, ), (1, ))
    assert_size_stride(arg115_1, (384, ), (1, ))
    assert_size_stride(arg116_1, (384, ), (1, ))
    assert_size_stride(arg117_1, (384, ), (1, ))
    assert_size_stride(arg118_1, (1408, ), (1, ))
    assert_size_stride(arg119_1, (1408, ), (1, ))
    assert_size_stride(arg120_1, (1408, ), (1, ))
    assert_size_stride(arg121_1, (1408, ), (1, ))
    assert_size_stride(arg122_1, (64, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(arg123_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg124_1, (96, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg125_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg126_1, (96, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg127_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg128_1, (192, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg129_1, (192, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg130_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg131_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg132_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg133_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg134_1, (192, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg135_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg136_1, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg137_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg138_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg139_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg140_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg141_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg142_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg143_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg144_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg145_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg146_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg147_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg148_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg149_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg150_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg151_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg152_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg153_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg154_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg155_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg156_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg157_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg158_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg159_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg160_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg161_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg162_1, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg163_1, (384, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg164_1, (1408, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg165_1, (1408, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg166_1, (1000, 1408), (1408, 1))
    assert_size_stride(arg167_1, (1000, ), (1, ))
    assert_size_stride(arg168_1, (64, ), (1, ))
    assert_size_stride(arg169_1, (64, ), (1, ))
    assert_size_stride(arg170_1, (64, ), (1, ))
    assert_size_stride(arg171_1, (64, ), (1, ))
    assert_size_stride(arg172_1, (96, ), (1, ))
    assert_size_stride(arg173_1, (96, ), (1, ))
    assert_size_stride(arg174_1, (96, ), (1, ))
    assert_size_stride(arg175_1, (96, ), (1, ))
    assert_size_stride(arg176_1, (96, ), (1, ))
    assert_size_stride(arg177_1, (96, ), (1, ))
    assert_size_stride(arg178_1, (96, ), (1, ))
    assert_size_stride(arg179_1, (96, ), (1, ))
    assert_size_stride(arg180_1, (96, ), (1, ))
    assert_size_stride(arg181_1, (96, ), (1, ))
    assert_size_stride(arg182_1, (192, ), (1, ))
    assert_size_stride(arg183_1, (192, ), (1, ))
    assert_size_stride(arg184_1, (192, ), (1, ))
    assert_size_stride(arg185_1, (192, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (192, ), (1, ))
    assert_size_stride(arg189_1, (192, ), (1, ))
    assert_size_stride(arg190_1, (192, ), (1, ))
    assert_size_stride(arg191_1, (192, ), (1, ))
    assert_size_stride(arg192_1, (192, ), (1, ))
    assert_size_stride(arg193_1, (192, ), (1, ))
    assert_size_stride(arg194_1, (192, ), (1, ))
    assert_size_stride(arg195_1, (192, ), (1, ))
    assert_size_stride(arg196_1, (192, ), (1, ))
    assert_size_stride(arg197_1, (192, ), (1, ))
    assert_size_stride(arg198_1, (192, ), (1, ))
    assert_size_stride(arg199_1, (192, ), (1, ))
    assert_size_stride(arg200_1, (192, ), (1, ))
    assert_size_stride(arg201_1, (192, ), (1, ))
    assert_size_stride(arg202_1, (192, ), (1, ))
    assert_size_stride(arg203_1, (192, ), (1, ))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (384, ), (1, ))
    assert_size_stride(arg206_1, (384, ), (1, ))
    assert_size_stride(arg207_1, (384, ), (1, ))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (384, ), (1, ))
    assert_size_stride(arg210_1, (384, ), (1, ))
    assert_size_stride(arg211_1, (384, ), (1, ))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (384, ), (1, ))
    assert_size_stride(arg214_1, (384, ), (1, ))
    assert_size_stride(arg215_1, (384, ), (1, ))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (384, ), (1, ))
    assert_size_stride(arg218_1, (384, ), (1, ))
    assert_size_stride(arg219_1, (384, ), (1, ))
    assert_size_stride(arg220_1, (384, ), (1, ))
    assert_size_stride(arg221_1, (384, ), (1, ))
    assert_size_stride(arg222_1, (384, ), (1, ))
    assert_size_stride(arg223_1, (384, ), (1, ))
    assert_size_stride(arg224_1, (384, ), (1, ))
    assert_size_stride(arg225_1, (384, ), (1, ))
    assert_size_stride(arg226_1, (384, ), (1, ))
    assert_size_stride(arg227_1, (384, ), (1, ))
    assert_size_stride(arg228_1, (384, ), (1, ))
    assert_size_stride(arg229_1, (384, ), (1, ))
    assert_size_stride(arg230_1, (384, ), (1, ))
    assert_size_stride(arg231_1, (384, ), (1, ))
    assert_size_stride(arg232_1, (384, ), (1, ))
    assert_size_stride(arg233_1, (384, ), (1, ))
    assert_size_stride(arg234_1, (384, ), (1, ))
    assert_size_stride(arg235_1, (384, ), (1, ))
    assert_size_stride(arg236_1, (384, ), (1, ))
    assert_size_stride(arg237_1, (384, ), (1, ))
    assert_size_stride(arg238_1, (384, ), (1, ))
    assert_size_stride(arg239_1, (384, ), (1, ))
    assert_size_stride(arg240_1, (384, ), (1, ))
    assert_size_stride(arg241_1, (384, ), (1, ))
    assert_size_stride(arg242_1, (384, ), (1, ))
    assert_size_stride(arg243_1, (384, ), (1, ))
    assert_size_stride(arg244_1, (384, ), (1, ))
    assert_size_stride(arg245_1, (384, ), (1, ))
    assert_size_stride(arg246_1, (384, ), (1, ))
    assert_size_stride(arg247_1, (384, ), (1, ))
    assert_size_stride(arg248_1, (384, ), (1, ))
    assert_size_stride(arg249_1, (384, ), (1, ))
    assert_size_stride(arg250_1, (384, ), (1, ))
    assert_size_stride(arg251_1, (384, ), (1, ))
    assert_size_stride(arg252_1, (384, ), (1, ))
    assert_size_stride(arg253_1, (384, ), (1, ))
    assert_size_stride(arg254_1, (384, ), (1, ))
    assert_size_stride(arg255_1, (384, ), (1, ))
    assert_size_stride(arg256_1, (384, ), (1, ))
    assert_size_stride(arg257_1, (384, ), (1, ))
    assert_size_stride(arg258_1, (384, ), (1, ))
    assert_size_stride(arg259_1, (384, ), (1, ))
    assert_size_stride(arg260_1, (384, ), (1, ))
    assert_size_stride(arg261_1, (384, ), (1, ))
    assert_size_stride(arg262_1, (384, ), (1, ))
    assert_size_stride(arg263_1, (384, ), (1, ))
    assert_size_stride(arg264_1, (384, ), (1, ))
    assert_size_stride(arg265_1, (384, ), (1, ))
    assert_size_stride(arg266_1, (384, ), (1, ))
    assert_size_stride(arg267_1, (384, ), (1, ))
    assert_size_stride(arg268_1, (384, ), (1, ))
    assert_size_stride(arg269_1, (384, ), (1, ))
    assert_size_stride(arg270_1, (384, ), (1, ))
    assert_size_stride(arg271_1, (384, ), (1, ))
    assert_size_stride(arg272_1, (384, ), (1, ))
    assert_size_stride(arg273_1, (384, ), (1, ))
    assert_size_stride(arg274_1, (384, ), (1, ))
    assert_size_stride(arg275_1, (384, ), (1, ))
    assert_size_stride(arg276_1, (384, ), (1, ))
    assert_size_stride(arg277_1, (384, ), (1, ))
    assert_size_stride(arg278_1, (384, ), (1, ))
    assert_size_stride(arg279_1, (384, ), (1, ))
    assert_size_stride(arg280_1, (384, ), (1, ))
    assert_size_stride(arg281_1, (384, ), (1, ))
    assert_size_stride(arg282_1, (384, ), (1, ))
    assert_size_stride(arg283_1, (384, ), (1, ))
    assert_size_stride(arg284_1, (384, ), (1, ))
    assert_size_stride(arg285_1, (384, ), (1, ))
    assert_size_stride(arg286_1, (1408, ), (1, ))
    assert_size_stride(arg287_1, (1408, ), (1, ))
    assert_size_stride(arg288_1, (1408, ), (1, ))
    assert_size_stride(arg289_1, (1408, ), (1, ))
    assert_size_stride(arg290_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_0(c_void_p(arg290_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf2.data_ptr()))
    del arg290_1
    # Source Nodes: [x], Original ATen: [aten.convolution]
    buf1 = extern_kernels.convolution(buf0, arg122_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf1, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del arg122_1
    del buf0
    buf3 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_1(c_void_p(arg123_1.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg123_1
    # Source Nodes: [x_5], Original ATen: [aten.convolution]
    buf4 = extern_kernels.convolution(buf2, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf4, (8, 64, 112, 112), (802816, 1, 7168, 64))
    del buf2
    del buf3
    buf5 = buf1; del buf1  # reuse
    buf6 = buf5; del buf5  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_2(c_void_p(buf6.data_ptr()), c_void_p(arg168_1.data_ptr()), c_void_p(arg169_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(arg170_1.data_ptr()), c_void_p(arg171_1.data_ptr()), c_void_p(arg2_1.data_ptr()), c_void_p(arg3_1.data_ptr()))
    del arg0_1
    del arg168_1
    del arg169_1
    del arg170_1
    del arg171_1
    del arg1_1
    del arg2_1
    del arg3_1
    del buf4
    # Source Nodes: [x_12, x_13], Original ATen: [aten.convolution, aten.relu]
    buf7 = extern_kernels.convolution(buf6, arg124_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf7, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del arg124_1
    buf8 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_3(c_void_p(arg125_1.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg125_1
    # Source Nodes: [x_18], Original ATen: [aten.convolution]
    buf9 = extern_kernels.convolution(buf6, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf9, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del buf6
    del buf8
    buf10 = buf7; del buf7  # reuse
    buf11 = buf10; del buf10  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_4(c_void_p(buf11.data_ptr()), c_void_p(arg172_1.data_ptr()), c_void_p(arg173_1.data_ptr()), c_void_p(arg4_1.data_ptr()), c_void_p(arg5_1.data_ptr()), c_void_p(buf9.data_ptr()), c_void_p(arg174_1.data_ptr()), c_void_p(arg175_1.data_ptr()), c_void_p(arg6_1.data_ptr()), c_void_p(arg7_1.data_ptr()))
    del arg172_1
    del arg173_1
    del arg174_1
    del arg175_1
    del arg4_1
    del arg5_1
    del arg6_1
    del arg7_1
    del buf9
    # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act, x_28], Original ATen: [aten.convolution, aten.relu]
    buf12 = extern_kernels.convolution(buf11, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf12, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del arg126_1
    buf13 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_5(c_void_p(arg127_1.data_ptr()), c_void_p(buf13.data_ptr()))
    del arg127_1
    # Source Nodes: [x_33], Original ATen: [aten.convolution]
    buf14 = extern_kernels.convolution(buf11, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf14, (8, 96, 56, 56), (301056, 1, 5376, 96))
    del buf13
    buf15 = buf12; del buf12  # reuse
    buf16 = buf11; del buf11  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_6(c_void_p(buf15.data_ptr()), c_void_p(buf16.data_ptr()), c_void_p(arg178_1.data_ptr()), c_void_p(arg179_1.data_ptr()), c_void_p(arg10_1.data_ptr()), c_void_p(arg11_1.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(arg180_1.data_ptr()), c_void_p(arg181_1.data_ptr()), c_void_p(arg12_1.data_ptr()), c_void_p(arg13_1.data_ptr()), c_void_p(arg176_1.data_ptr()), c_void_p(arg177_1.data_ptr()), c_void_p(arg8_1.data_ptr()), c_void_p(arg9_1.data_ptr()))
    del arg10_1
    del arg11_1
    del arg12_1
    del arg13_1
    del arg176_1
    del arg177_1
    del arg178_1
    del arg179_1
    del arg180_1
    del arg181_1
    del arg8_1
    del arg9_1
    del buf14
    del buf15
    # Source Nodes: [x_42], Original ATen: [aten.convolution]
    buf17 = extern_kernels.convolution(buf16, arg128_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf17, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg128_1
    buf18 = empty_strided((192, 96, 3, 3), (864, 1, 288, 96), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_7(c_void_p(arg129_1.data_ptr()), c_void_p(buf18.data_ptr()))
    del arg129_1
    # Source Nodes: [x_47], Original ATen: [aten.convolution]
    buf19 = extern_kernels.convolution(buf16, buf18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf19, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del buf16
    del buf18
    buf20 = buf17; del buf17  # reuse
    buf21 = buf20; del buf20  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_8(c_void_p(buf21.data_ptr()), c_void_p(arg182_1.data_ptr()), c_void_p(arg183_1.data_ptr()), c_void_p(arg14_1.data_ptr()), c_void_p(arg15_1.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(arg184_1.data_ptr()), c_void_p(arg185_1.data_ptr()), c_void_p(arg16_1.data_ptr()), c_void_p(arg17_1.data_ptr()))
    del arg14_1
    del arg15_1
    del arg16_1
    del arg17_1
    del arg182_1
    del arg183_1
    del arg184_1
    del arg185_1
    del buf19
    # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act, x_57], Original ATen: [aten.convolution, aten.relu]
    buf22 = extern_kernels.convolution(buf21, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf22, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg130_1
    buf23 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_9(c_void_p(arg131_1.data_ptr()), c_void_p(buf23.data_ptr()))
    del arg131_1
    # Source Nodes: [x_62], Original ATen: [aten.convolution]
    buf24 = extern_kernels.convolution(buf21, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf24, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf25 = buf22; del buf22  # reuse
    buf26 = buf21; del buf21  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_10(c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()), c_void_p(arg188_1.data_ptr()), c_void_p(arg189_1.data_ptr()), c_void_p(arg20_1.data_ptr()), c_void_p(arg21_1.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(arg190_1.data_ptr()), c_void_p(arg191_1.data_ptr()), c_void_p(arg22_1.data_ptr()), c_void_p(arg23_1.data_ptr()), c_void_p(arg186_1.data_ptr()), c_void_p(arg187_1.data_ptr()), c_void_p(arg18_1.data_ptr()), c_void_p(arg19_1.data_ptr()))
    del arg186_1
    del arg187_1
    del arg188_1
    del arg189_1
    del arg18_1
    del arg190_1
    del arg191_1
    del arg19_1
    del arg20_1
    del arg21_1
    del arg22_1
    del arg23_1
    del buf24
    del buf25
    # Source Nodes: [x_74], Original ATen: [aten.convolution]
    buf27 = extern_kernels.convolution(buf26, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf27, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg132_1
    buf28 = buf23; del buf23  # reuse
    cpp_fused_convolution_11(c_void_p(arg133_1.data_ptr()), c_void_p(buf28.data_ptr()))
    del arg133_1
    # Source Nodes: [x_79], Original ATen: [aten.convolution]
    buf29 = extern_kernels.convolution(buf26, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf29, (8, 192, 28, 28), (150528, 1, 5376, 192))
    buf30 = buf27; del buf27  # reuse
    buf31 = buf26; del buf26  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_12(c_void_p(buf30.data_ptr()), c_void_p(buf31.data_ptr()), c_void_p(arg194_1.data_ptr()), c_void_p(arg195_1.data_ptr()), c_void_p(arg26_1.data_ptr()), c_void_p(arg27_1.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(arg196_1.data_ptr()), c_void_p(arg197_1.data_ptr()), c_void_p(arg28_1.data_ptr()), c_void_p(arg29_1.data_ptr()), c_void_p(arg192_1.data_ptr()), c_void_p(arg193_1.data_ptr()), c_void_p(arg24_1.data_ptr()), c_void_p(arg25_1.data_ptr()))
    del arg192_1
    del arg193_1
    del arg194_1
    del arg195_1
    del arg196_1
    del arg197_1
    del arg24_1
    del arg25_1
    del arg26_1
    del arg27_1
    del arg28_1
    del arg29_1
    del buf29
    del buf30
    # Source Nodes: [x_91], Original ATen: [aten.convolution]
    buf32 = extern_kernels.convolution(buf31, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf32, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del arg134_1
    buf33 = buf28; del buf28  # reuse
    cpp_fused_convolution_13(c_void_p(arg135_1.data_ptr()), c_void_p(buf33.data_ptr()))
    del arg135_1
    # Source Nodes: [x_96], Original ATen: [aten.convolution]
    buf34 = extern_kernels.convolution(buf31, buf33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf34, (8, 192, 28, 28), (150528, 1, 5376, 192))
    del buf33
    buf35 = buf32; del buf32  # reuse
    buf36 = buf31; del buf31  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_14(c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(arg200_1.data_ptr()), c_void_p(arg201_1.data_ptr()), c_void_p(arg32_1.data_ptr()), c_void_p(arg33_1.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(arg202_1.data_ptr()), c_void_p(arg203_1.data_ptr()), c_void_p(arg34_1.data_ptr()), c_void_p(arg35_1.data_ptr()), c_void_p(arg198_1.data_ptr()), c_void_p(arg199_1.data_ptr()), c_void_p(arg30_1.data_ptr()), c_void_p(arg31_1.data_ptr()))
    del arg198_1
    del arg199_1
    del arg200_1
    del arg201_1
    del arg202_1
    del arg203_1
    del arg30_1
    del arg31_1
    del arg32_1
    del arg33_1
    del arg34_1
    del arg35_1
    del buf34
    del buf35
    # Source Nodes: [x_105], Original ATen: [aten.convolution]
    buf37 = extern_kernels.convolution(buf36, arg136_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf37, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg136_1
    buf38 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_15(c_void_p(arg137_1.data_ptr()), c_void_p(buf38.data_ptr()))
    del arg137_1
    # Source Nodes: [x_110], Original ATen: [aten.convolution]
    buf39 = extern_kernels.convolution(buf36, buf38, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf39, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del buf36
    del buf38
    buf40 = buf37; del buf37  # reuse
    buf41 = buf40; del buf40  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_16(c_void_p(buf41.data_ptr()), c_void_p(arg204_1.data_ptr()), c_void_p(arg205_1.data_ptr()), c_void_p(arg36_1.data_ptr()), c_void_p(arg37_1.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(arg206_1.data_ptr()), c_void_p(arg207_1.data_ptr()), c_void_p(arg38_1.data_ptr()), c_void_p(arg39_1.data_ptr()))
    del arg204_1
    del arg205_1
    del arg206_1
    del arg207_1
    del arg36_1
    del arg37_1
    del arg38_1
    del arg39_1
    del buf39
    # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act, x_120], Original ATen: [aten.convolution, aten.relu]
    buf42 = extern_kernels.convolution(buf41, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf42, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg138_1
    buf43 = empty_strided((384, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_17(c_void_p(arg139_1.data_ptr()), c_void_p(buf43.data_ptr()))
    del arg139_1
    # Source Nodes: [x_125], Original ATen: [aten.convolution]
    buf44 = extern_kernels.convolution(buf41, buf43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf44, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf45 = buf42; del buf42  # reuse
    buf46 = buf41; del buf41  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_18(c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(arg210_1.data_ptr()), c_void_p(arg211_1.data_ptr()), c_void_p(arg42_1.data_ptr()), c_void_p(arg43_1.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(arg212_1.data_ptr()), c_void_p(arg213_1.data_ptr()), c_void_p(arg44_1.data_ptr()), c_void_p(arg45_1.data_ptr()), c_void_p(arg208_1.data_ptr()), c_void_p(arg209_1.data_ptr()), c_void_p(arg40_1.data_ptr()), c_void_p(arg41_1.data_ptr()))
    del arg208_1
    del arg209_1
    del arg210_1
    del arg211_1
    del arg212_1
    del arg213_1
    del arg40_1
    del arg41_1
    del arg42_1
    del arg43_1
    del arg44_1
    del arg45_1
    del buf44
    del buf45
    # Source Nodes: [x_137], Original ATen: [aten.convolution]
    buf47 = extern_kernels.convolution(buf46, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf47, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg140_1
    buf48 = buf43; del buf43  # reuse
    cpp_fused_convolution_19(c_void_p(arg141_1.data_ptr()), c_void_p(buf48.data_ptr()))
    del arg141_1
    # Source Nodes: [x_142], Original ATen: [aten.convolution]
    buf49 = extern_kernels.convolution(buf46, buf48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf49, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf50 = buf47; del buf47  # reuse
    buf51 = buf46; del buf46  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_20(c_void_p(buf50.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(arg216_1.data_ptr()), c_void_p(arg217_1.data_ptr()), c_void_p(arg48_1.data_ptr()), c_void_p(arg49_1.data_ptr()), c_void_p(buf49.data_ptr()), c_void_p(arg218_1.data_ptr()), c_void_p(arg219_1.data_ptr()), c_void_p(arg50_1.data_ptr()), c_void_p(arg51_1.data_ptr()), c_void_p(arg214_1.data_ptr()), c_void_p(arg215_1.data_ptr()), c_void_p(arg46_1.data_ptr()), c_void_p(arg47_1.data_ptr()))
    del arg214_1
    del arg215_1
    del arg216_1
    del arg217_1
    del arg218_1
    del arg219_1
    del arg46_1
    del arg47_1
    del arg48_1
    del arg49_1
    del arg50_1
    del arg51_1
    del buf49
    del buf50
    # Source Nodes: [x_154], Original ATen: [aten.convolution]
    buf52 = extern_kernels.convolution(buf51, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf52, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg142_1
    buf53 = buf48; del buf48  # reuse
    cpp_fused_convolution_21(c_void_p(arg143_1.data_ptr()), c_void_p(buf53.data_ptr()))
    del arg143_1
    # Source Nodes: [x_159], Original ATen: [aten.convolution]
    buf54 = extern_kernels.convolution(buf51, buf53, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf54, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf55 = buf52; del buf52  # reuse
    buf56 = buf51; del buf51  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_22(c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(arg222_1.data_ptr()), c_void_p(arg223_1.data_ptr()), c_void_p(arg54_1.data_ptr()), c_void_p(arg55_1.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(arg224_1.data_ptr()), c_void_p(arg225_1.data_ptr()), c_void_p(arg56_1.data_ptr()), c_void_p(arg57_1.data_ptr()), c_void_p(arg220_1.data_ptr()), c_void_p(arg221_1.data_ptr()), c_void_p(arg52_1.data_ptr()), c_void_p(arg53_1.data_ptr()))
    del arg220_1
    del arg221_1
    del arg222_1
    del arg223_1
    del arg224_1
    del arg225_1
    del arg52_1
    del arg53_1
    del arg54_1
    del arg55_1
    del arg56_1
    del arg57_1
    del buf54
    del buf55
    # Source Nodes: [x_171], Original ATen: [aten.convolution]
    buf57 = extern_kernels.convolution(buf56, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf57, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg144_1
    buf58 = buf53; del buf53  # reuse
    cpp_fused_convolution_23(c_void_p(arg145_1.data_ptr()), c_void_p(buf58.data_ptr()))
    del arg145_1
    # Source Nodes: [x_176], Original ATen: [aten.convolution]
    buf59 = extern_kernels.convolution(buf56, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf59, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf60 = buf57; del buf57  # reuse
    buf61 = buf56; del buf56  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_24(c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(arg228_1.data_ptr()), c_void_p(arg229_1.data_ptr()), c_void_p(arg60_1.data_ptr()), c_void_p(arg61_1.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(arg230_1.data_ptr()), c_void_p(arg231_1.data_ptr()), c_void_p(arg62_1.data_ptr()), c_void_p(arg63_1.data_ptr()), c_void_p(arg226_1.data_ptr()), c_void_p(arg227_1.data_ptr()), c_void_p(arg58_1.data_ptr()), c_void_p(arg59_1.data_ptr()))
    del arg226_1
    del arg227_1
    del arg228_1
    del arg229_1
    del arg230_1
    del arg231_1
    del arg58_1
    del arg59_1
    del arg60_1
    del arg61_1
    del arg62_1
    del arg63_1
    del buf59
    del buf60
    # Source Nodes: [x_188], Original ATen: [aten.convolution]
    buf62 = extern_kernels.convolution(buf61, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf62, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg146_1
    buf63 = buf58; del buf58  # reuse
    cpp_fused_convolution_25(c_void_p(arg147_1.data_ptr()), c_void_p(buf63.data_ptr()))
    del arg147_1
    # Source Nodes: [x_193], Original ATen: [aten.convolution]
    buf64 = extern_kernels.convolution(buf61, buf63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf64, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf65 = buf62; del buf62  # reuse
    buf66 = buf61; del buf61  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_26(c_void_p(buf65.data_ptr()), c_void_p(buf66.data_ptr()), c_void_p(arg234_1.data_ptr()), c_void_p(arg235_1.data_ptr()), c_void_p(arg66_1.data_ptr()), c_void_p(arg67_1.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(arg236_1.data_ptr()), c_void_p(arg237_1.data_ptr()), c_void_p(arg68_1.data_ptr()), c_void_p(arg69_1.data_ptr()), c_void_p(arg232_1.data_ptr()), c_void_p(arg233_1.data_ptr()), c_void_p(arg64_1.data_ptr()), c_void_p(arg65_1.data_ptr()))
    del arg232_1
    del arg233_1
    del arg234_1
    del arg235_1
    del arg236_1
    del arg237_1
    del arg64_1
    del arg65_1
    del arg66_1
    del arg67_1
    del arg68_1
    del arg69_1
    del buf64
    del buf65
    # Source Nodes: [x_205], Original ATen: [aten.convolution]
    buf67 = extern_kernels.convolution(buf66, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf67, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg148_1
    buf68 = buf63; del buf63  # reuse
    cpp_fused_convolution_27(c_void_p(arg149_1.data_ptr()), c_void_p(buf68.data_ptr()))
    del arg149_1
    # Source Nodes: [x_210], Original ATen: [aten.convolution]
    buf69 = extern_kernels.convolution(buf66, buf68, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf69, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf70 = buf67; del buf67  # reuse
    buf71 = buf66; del buf66  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_28(c_void_p(buf70.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(arg240_1.data_ptr()), c_void_p(arg241_1.data_ptr()), c_void_p(arg72_1.data_ptr()), c_void_p(arg73_1.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(arg242_1.data_ptr()), c_void_p(arg243_1.data_ptr()), c_void_p(arg74_1.data_ptr()), c_void_p(arg75_1.data_ptr()), c_void_p(arg238_1.data_ptr()), c_void_p(arg239_1.data_ptr()), c_void_p(arg70_1.data_ptr()), c_void_p(arg71_1.data_ptr()))
    del arg238_1
    del arg239_1
    del arg240_1
    del arg241_1
    del arg242_1
    del arg243_1
    del arg70_1
    del arg71_1
    del arg72_1
    del arg73_1
    del arg74_1
    del arg75_1
    del buf69
    del buf70
    # Source Nodes: [x_222], Original ATen: [aten.convolution]
    buf72 = extern_kernels.convolution(buf71, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf72, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg150_1
    buf73 = buf68; del buf68  # reuse
    cpp_fused_convolution_29(c_void_p(arg151_1.data_ptr()), c_void_p(buf73.data_ptr()))
    del arg151_1
    # Source Nodes: [x_227], Original ATen: [aten.convolution]
    buf74 = extern_kernels.convolution(buf71, buf73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf74, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf75 = buf72; del buf72  # reuse
    buf76 = buf71; del buf71  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_30(c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(arg246_1.data_ptr()), c_void_p(arg247_1.data_ptr()), c_void_p(arg78_1.data_ptr()), c_void_p(arg79_1.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(arg248_1.data_ptr()), c_void_p(arg249_1.data_ptr()), c_void_p(arg80_1.data_ptr()), c_void_p(arg81_1.data_ptr()), c_void_p(arg244_1.data_ptr()), c_void_p(arg245_1.data_ptr()), c_void_p(arg76_1.data_ptr()), c_void_p(arg77_1.data_ptr()))
    del arg244_1
    del arg245_1
    del arg246_1
    del arg247_1
    del arg248_1
    del arg249_1
    del arg76_1
    del arg77_1
    del arg78_1
    del arg79_1
    del arg80_1
    del arg81_1
    del buf74
    del buf75
    # Source Nodes: [x_239], Original ATen: [aten.convolution]
    buf77 = extern_kernels.convolution(buf76, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf77, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg152_1
    buf78 = buf73; del buf73  # reuse
    cpp_fused_convolution_31(c_void_p(arg153_1.data_ptr()), c_void_p(buf78.data_ptr()))
    del arg153_1
    # Source Nodes: [x_244], Original ATen: [aten.convolution]
    buf79 = extern_kernels.convolution(buf76, buf78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf79, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf80 = buf77; del buf77  # reuse
    buf81 = buf76; del buf76  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_32(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(arg252_1.data_ptr()), c_void_p(arg253_1.data_ptr()), c_void_p(arg84_1.data_ptr()), c_void_p(arg85_1.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(arg254_1.data_ptr()), c_void_p(arg255_1.data_ptr()), c_void_p(arg86_1.data_ptr()), c_void_p(arg87_1.data_ptr()), c_void_p(arg250_1.data_ptr()), c_void_p(arg251_1.data_ptr()), c_void_p(arg82_1.data_ptr()), c_void_p(arg83_1.data_ptr()))
    del arg250_1
    del arg251_1
    del arg252_1
    del arg253_1
    del arg254_1
    del arg255_1
    del arg82_1
    del arg83_1
    del arg84_1
    del arg85_1
    del arg86_1
    del arg87_1
    del buf79
    del buf80
    # Source Nodes: [x_256], Original ATen: [aten.convolution]
    buf82 = extern_kernels.convolution(buf81, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf82, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg154_1
    buf83 = buf78; del buf78  # reuse
    cpp_fused_convolution_33(c_void_p(arg155_1.data_ptr()), c_void_p(buf83.data_ptr()))
    del arg155_1
    # Source Nodes: [x_261], Original ATen: [aten.convolution]
    buf84 = extern_kernels.convolution(buf81, buf83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf84, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf85 = buf82; del buf82  # reuse
    buf86 = buf81; del buf81  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_34(c_void_p(buf85.data_ptr()), c_void_p(buf86.data_ptr()), c_void_p(arg258_1.data_ptr()), c_void_p(arg259_1.data_ptr()), c_void_p(arg90_1.data_ptr()), c_void_p(arg91_1.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(arg260_1.data_ptr()), c_void_p(arg261_1.data_ptr()), c_void_p(arg92_1.data_ptr()), c_void_p(arg93_1.data_ptr()), c_void_p(arg256_1.data_ptr()), c_void_p(arg257_1.data_ptr()), c_void_p(arg88_1.data_ptr()), c_void_p(arg89_1.data_ptr()))
    del arg256_1
    del arg257_1
    del arg258_1
    del arg259_1
    del arg260_1
    del arg261_1
    del arg88_1
    del arg89_1
    del arg90_1
    del arg91_1
    del arg92_1
    del arg93_1
    del buf84
    del buf85
    # Source Nodes: [x_273], Original ATen: [aten.convolution]
    buf87 = extern_kernels.convolution(buf86, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf87, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg156_1
    buf88 = buf83; del buf83  # reuse
    cpp_fused_convolution_35(c_void_p(arg157_1.data_ptr()), c_void_p(buf88.data_ptr()))
    del arg157_1
    # Source Nodes: [x_278], Original ATen: [aten.convolution]
    buf89 = extern_kernels.convolution(buf86, buf88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf89, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf90 = buf87; del buf87  # reuse
    buf91 = buf86; del buf86  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_36(c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(arg264_1.data_ptr()), c_void_p(arg265_1.data_ptr()), c_void_p(arg96_1.data_ptr()), c_void_p(arg97_1.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(arg266_1.data_ptr()), c_void_p(arg267_1.data_ptr()), c_void_p(arg98_1.data_ptr()), c_void_p(arg99_1.data_ptr()), c_void_p(arg262_1.data_ptr()), c_void_p(arg263_1.data_ptr()), c_void_p(arg94_1.data_ptr()), c_void_p(arg95_1.data_ptr()))
    del arg262_1
    del arg263_1
    del arg264_1
    del arg265_1
    del arg266_1
    del arg267_1
    del arg94_1
    del arg95_1
    del arg96_1
    del arg97_1
    del arg98_1
    del arg99_1
    del buf89
    del buf90
    # Source Nodes: [x_290], Original ATen: [aten.convolution]
    buf92 = extern_kernels.convolution(buf91, arg158_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf92, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg158_1
    buf93 = buf88; del buf88  # reuse
    cpp_fused_convolution_37(c_void_p(arg159_1.data_ptr()), c_void_p(buf93.data_ptr()))
    del arg159_1
    # Source Nodes: [x_295], Original ATen: [aten.convolution]
    buf94 = extern_kernels.convolution(buf91, buf93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf94, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf95 = buf92; del buf92  # reuse
    buf96 = buf91; del buf91  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_38(c_void_p(buf95.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg270_1.data_ptr()), c_void_p(arg271_1.data_ptr()), c_void_p(arg102_1.data_ptr()), c_void_p(arg103_1.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(arg272_1.data_ptr()), c_void_p(arg273_1.data_ptr()), c_void_p(arg104_1.data_ptr()), c_void_p(arg105_1.data_ptr()), c_void_p(arg268_1.data_ptr()), c_void_p(arg269_1.data_ptr()), c_void_p(arg100_1.data_ptr()), c_void_p(arg101_1.data_ptr()))
    del arg100_1
    del arg101_1
    del arg102_1
    del arg103_1
    del arg104_1
    del arg105_1
    del arg268_1
    del arg269_1
    del arg270_1
    del arg271_1
    del arg272_1
    del arg273_1
    del buf94
    del buf95
    # Source Nodes: [x_307], Original ATen: [aten.convolution]
    buf97 = extern_kernels.convolution(buf96, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf97, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg160_1
    buf98 = buf93; del buf93  # reuse
    cpp_fused_convolution_39(c_void_p(arg161_1.data_ptr()), c_void_p(buf98.data_ptr()))
    del arg161_1
    # Source Nodes: [x_312], Original ATen: [aten.convolution]
    buf99 = extern_kernels.convolution(buf96, buf98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf99, (8, 384, 14, 14), (75264, 1, 5376, 384))
    buf100 = buf97; del buf97  # reuse
    buf101 = buf100; del buf100  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_40(c_void_p(buf101.data_ptr()), c_void_p(arg276_1.data_ptr()), c_void_p(arg277_1.data_ptr()), c_void_p(arg108_1.data_ptr()), c_void_p(arg109_1.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(arg278_1.data_ptr()), c_void_p(arg279_1.data_ptr()), c_void_p(arg110_1.data_ptr()), c_void_p(arg111_1.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(arg274_1.data_ptr()), c_void_p(arg275_1.data_ptr()), c_void_p(arg106_1.data_ptr()), c_void_p(arg107_1.data_ptr()))
    del arg106_1
    del arg107_1
    del arg108_1
    del arg109_1
    del arg110_1
    del arg111_1
    del arg274_1
    del arg275_1
    del arg276_1
    del arg277_1
    del arg278_1
    del arg279_1
    del buf96
    del buf99
    # Source Nodes: [x_324], Original ATen: [aten.convolution]
    buf102 = extern_kernels.convolution(buf101, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf102, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del arg162_1
    buf103 = buf98; del buf98  # reuse
    cpp_fused_convolution_41(c_void_p(arg163_1.data_ptr()), c_void_p(buf103.data_ptr()))
    del arg163_1
    # Source Nodes: [x_329], Original ATen: [aten.convolution]
    buf104 = extern_kernels.convolution(buf101, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf104, (8, 384, 14, 14), (75264, 1, 5376, 384))
    del buf103
    buf105 = buf102; del buf102  # reuse
    buf106 = buf101; del buf101  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_relu_42(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(arg282_1.data_ptr()), c_void_p(arg283_1.data_ptr()), c_void_p(arg114_1.data_ptr()), c_void_p(arg115_1.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(arg284_1.data_ptr()), c_void_p(arg285_1.data_ptr()), c_void_p(arg116_1.data_ptr()), c_void_p(arg117_1.data_ptr()), c_void_p(arg280_1.data_ptr()), c_void_p(arg281_1.data_ptr()), c_void_p(arg112_1.data_ptr()), c_void_p(arg113_1.data_ptr()))
    del arg112_1
    del arg113_1
    del arg114_1
    del arg115_1
    del arg116_1
    del arg117_1
    del arg280_1
    del arg281_1
    del arg282_1
    del arg283_1
    del arg284_1
    del arg285_1
    del buf104
    del buf105
    # Source Nodes: [x_338], Original ATen: [aten.convolution]
    buf107 = extern_kernels.convolution(buf106, arg164_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf107, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    del arg164_1
    buf108 = empty_strided((1408, 384, 3, 3), (3456, 1, 1152, 384), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_43(c_void_p(arg165_1.data_ptr()), c_void_p(buf108.data_ptr()))
    del arg165_1
    # Source Nodes: [x_343], Original ATen: [aten.convolution]
    buf109 = extern_kernels.convolution(buf106, buf108, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
    assert_size_stride(buf109, (8, 1408, 7, 7), (68992, 1, 9856, 1408))
    del buf106
    del buf108
    buf110 = buf107; del buf107  # reuse
    buf111 = empty_strided((8, 1408, 1, 1), (1408, 1, 11264, 11264), device='cpu', dtype=torch.float32)
    buf112 = reinterpret_tensor(buf111, (8, 1408, 1, 1), (1408, 1, 1, 1), 0); del buf111  # reuse
    cpp_fused__native_batch_norm_legit_no_training_add_mean_relu_44(c_void_p(buf110.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(arg286_1.data_ptr()), c_void_p(arg287_1.data_ptr()), c_void_p(arg118_1.data_ptr()), c_void_p(arg119_1.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(arg288_1.data_ptr()), c_void_p(arg289_1.data_ptr()), c_void_p(arg120_1.data_ptr()), c_void_p(arg121_1.data_ptr()))
    del arg118_1
    del arg119_1
    del arg120_1
    del arg121_1
    del arg286_1
    del arg287_1
    del arg288_1
    del arg289_1
    del buf109
    del buf110
    buf113 = empty((8, 1000), device='cpu', dtype=torch.float32)
    # Source Nodes: [x_357], Original ATen: [aten.addmm]
    extern_kernels.addmm(arg167_1, reinterpret_tensor(buf112, (8, 1408), (1408, 1), 0), reinterpret_tensor(arg166_1, (1408, 1000), (1, 1408), 0), alpha=1, beta=1, out=buf113)
    del arg166_1
    del arg167_1
    return (buf113, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg4_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg5_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg6_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg7_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg8_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg11_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg12_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg13_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg14_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg15_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg16_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg17_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg18_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg19_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg20_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg21_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg22_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg23_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg24_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg25_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg26_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg36_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg37_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg38_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg39_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg40_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg41_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg42_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg43_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg44_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg45_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg46_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg47_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg48_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg49_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg50_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg51_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg54_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg55_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg56_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg57_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg58_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg59_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg60_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg61_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg63_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg64_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg66_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg67_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg68_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg69_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg70_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg72_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg73_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg75_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg76_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg78_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg79_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg80_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg81_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg82_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg84_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg85_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg87_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg88_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg92_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg93_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg94_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg96_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg97_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg99_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg100_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg102_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg103_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg104_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg105_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg106_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg108_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg109_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg111_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg112_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg114_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg115_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg116_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg117_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg118_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg119_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg120_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg121_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg122_1 = rand_strided((64, 3, 1, 1), (3, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg123_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg124_1 = rand_strided((96, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg125_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg126_1 = rand_strided((96, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg127_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg128_1 = rand_strided((192, 96, 1, 1), (96, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg129_1 = rand_strided((192, 96, 3, 3), (864, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg130_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg131_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg132_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg133_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg134_1 = rand_strided((192, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg135_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg136_1 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg137_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg138_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg139_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg140_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg141_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg142_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg143_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg144_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg145_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg146_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg147_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg148_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg149_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg150_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg151_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg152_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg153_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg154_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg155_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg156_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg157_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg158_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg159_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg160_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg161_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg162_1 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg163_1 = rand_strided((384, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg164_1 = rand_strided((1408, 384, 1, 1), (384, 1, 1, 1), device='cpu', dtype=torch.float32)
    arg165_1 = rand_strided((1408, 384, 3, 3), (3456, 9, 3, 1), device='cpu', dtype=torch.float32)
    arg166_1 = rand_strided((1000, 1408), (1408, 1), device='cpu', dtype=torch.float32)
    arg167_1 = rand_strided((1000, ), (1, ), device='cpu', dtype=torch.float32)
    arg168_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg169_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg170_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg171_1 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    arg172_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg173_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg174_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg175_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg176_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg177_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg178_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg179_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg180_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg181_1 = rand_strided((96, ), (1, ), device='cpu', dtype=torch.float32)
    arg182_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg183_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg184_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg185_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg188_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg189_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg190_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg191_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg192_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg193_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg194_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg195_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg196_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg197_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg198_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg199_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg200_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg201_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg202_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg203_1 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg205_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg206_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg207_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg209_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg210_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg211_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg213_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg214_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg215_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg217_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg218_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg219_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg220_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg221_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg222_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg223_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg224_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg225_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg226_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg227_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg228_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg229_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg230_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg231_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg232_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg233_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg234_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg235_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg236_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg237_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg238_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg239_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg240_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg241_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg242_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg243_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg244_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg245_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg246_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg247_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg248_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg249_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg250_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg251_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg252_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg253_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg254_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg255_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg256_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg257_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg258_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg259_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg260_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg261_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg262_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg263_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg264_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg265_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg266_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg267_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg268_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg269_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg270_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg271_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg272_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg273_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg274_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg275_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg276_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg277_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg278_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg279_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg280_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg281_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg282_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg283_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg284_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg285_1 = rand_strided((384, ), (1, ), device='cpu', dtype=torch.float32)
    arg286_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg287_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg288_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg289_1 = rand_strided((1408, ), (1, ), device='cpu', dtype=torch.float32)
    arg290_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('repvgg_a2', benchmark_compiled_module)
