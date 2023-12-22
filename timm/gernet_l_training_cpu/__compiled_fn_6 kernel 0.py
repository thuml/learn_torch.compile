
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const bool* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1000L*x1)));
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
            }
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(64L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2560L*x2) + (163840L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2560L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2560L*x2) + (163840L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(64.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp11 = tmp7 * tmp10;
                            tmp_acc0_vec = tmp_acc0_vec + tmp7;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2560L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2560L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2560L*x1) + (163840L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2560L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2560L*x1) + (163840L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(64.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp12 = static_cast<float>(0.001953125);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp16 = tmp15 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp10 * tmp17;
                        auto tmp19 = tmp7 - tmp18;
                        auto tmp21 = tmp20 * tmp13;
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = tmp15 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        tmp25.store(out_ptr4 + static_cast<long>(x2 + (2560L*x1) + (163840L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4 = async_compile.cpp('''
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
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = to_float_mask(tmp4 <= tmp2);
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp5);
                        auto tmp9 = tmp7 + tmp8;
                        auto tmp10 = decltype(tmp2)::blendv(tmp9, tmp2, tmp3);
                        auto tmp13 = tmp11 - tmp12;
                        auto tmp14 = tmp10 * tmp13;
                        tmp_acc0_vec = tmp_acc0_vec + tmp10;
                        tmp_acc1_vec = tmp_acc1_vec + tmp14;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = to_float_mask(tmp4 <= tmp2);
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp5);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = decltype(tmp2)::blendv(tmp9, tmp2, tmp3);
                    auto tmp13 = tmp11 - tmp12;
                    auto tmp15 = static_cast<float>(0.001953125);
                    auto tmp16 = at::vec::Vectorized<float>(tmp15);
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp19 = tmp18 * tmp18;
                    auto tmp20 = tmp17 * tmp19;
                    auto tmp21 = tmp13 * tmp20;
                    auto tmp22 = tmp10 - tmp21;
                    auto tmp24 = tmp23 * tmp16;
                    auto tmp25 = tmp22 - tmp24;
                    auto tmp27 = tmp18 * tmp26;
                    auto tmp28 = tmp25 * tmp27;
                    tmp28.store(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_7 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp7 = to_float_mask(tmp6 <= tmp2);
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp7);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp5);
                auto tmp14 = tmp12 + tmp13;
                auto tmp15 = decltype(tmp2)::blendv(tmp14, tmp2, tmp3);
                tmp15.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001953125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.001953125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001953125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_16 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.001953125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001953125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_22 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.001953125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(327680L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.001953125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(512L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(512L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.001953125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (1920L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1920L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (1920L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (1920L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_29 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_35 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38 = async_compile.cpp('''
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
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    tmp18.store(out_ptr3 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.00048828125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1310720L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (640L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(640L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (640L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.00048828125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (640L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (640L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(640L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(2048L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.00048828125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (160L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_48 = async_compile.cpp('''
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
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(0.0001220703125);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(out_ptr2 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1572864L); x0+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp1 = static_cast<float>(0.0);
                auto tmp2 = at::vec::Vectorized<float>(tmp1);
                auto tmp3 = to_float_mask(tmp0 <= tmp2);
                auto tmp5 = to_float_mask(tmp4 <= tmp2);
                auto tmp8 = tmp6 + tmp7;
                auto tmp9 = decltype(tmp2)::blendv(tmp8, tmp2, tmp5);
                auto tmp11 = tmp9 + tmp10;
                auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                tmp12.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp7 = tmp5 - tmp6;
                        auto tmp8 = tmp0 * tmp7;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp30 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp5 = static_cast<float>(0.0001220703125);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = tmp4 * tmp6;
                    auto tmp9 = tmp8 * tmp8;
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp3 * tmp10;
                    auto tmp12 = tmp0 - tmp11;
                    auto tmp14 = tmp13 * tmp6;
                    auto tmp15 = tmp12 - tmp14;
                    auto tmp17 = tmp8 * tmp16;
                    auto tmp18 = tmp15 * tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp23 = tmp22 * tmp6;
                    auto tmp25 = tmp24 * tmp24;
                    auto tmp26 = tmp23 * tmp25;
                    auto tmp27 = tmp21 * tmp26;
                    auto tmp28 = tmp0 - tmp27;
                    auto tmp29 = tmp28 - tmp14;
                    auto tmp31 = tmp24 * tmp30;
                    auto tmp32 = tmp29 * tmp31;
                    tmp18.store(out_ptr4 + static_cast<long>(x1 + (192L*x0)));
                    tmp32.store(out_ptr5 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (192L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(192L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(192L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (192L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(0.0001220703125);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (192L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_53 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                        tmp_acc2_vec = tmp_acc2_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp26 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp27 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp29 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x1));
                    auto tmp31 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp37 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(3.0517578125e-05);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    auto tmp28 = tmp26 - tmp27;
                    auto tmp30 = tmp29 * tmp13;
                    auto tmp32 = tmp31 * tmp31;
                    auto tmp33 = tmp30 * tmp32;
                    auto tmp34 = tmp28 * tmp33;
                    auto tmp35 = tmp7 - tmp34;
                    auto tmp36 = tmp35 - tmp21;
                    auto tmp38 = tmp31 * tmp37;
                    auto tmp39 = tmp36 * tmp38;
                    tmp25.store(out_ptr3 + static_cast<long>(x1 + (128L*x0)));
                    tmp39.store(out_ptr4 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = tmp0 * tmp1;
            tmp2.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(32768L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (128L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                        auto tmp8 = tmp6 - tmp7;
                        auto tmp9 = tmp5 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                        tmp_acc1_vec = tmp_acc1_vec + tmp9;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32768L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(128L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (128L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp18 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp8 = tmp6 - tmp7;
                    auto tmp10 = static_cast<float>(3.0517578125e-05);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp9 * tmp11;
                    auto tmp14 = tmp13 * tmp13;
                    auto tmp15 = tmp12 * tmp14;
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp17 = tmp5 - tmp16;
                    auto tmp19 = tmp18 * tmp11;
                    auto tmp20 = tmp17 - tmp19;
                    auto tmp22 = tmp13 * tmp21;
                    auto tmp23 = tmp20 * tmp22;
                    tmp23.store(in_out_ptr0 + static_cast<long>(x1 + (128L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       float* out_ptr0)
{
    auto in_ptr1 = in_out_ptr0;
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(131072L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (32L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                        auto tmp10 = tmp8 - tmp9;
                        auto tmp11 = tmp7 * tmp10;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                        tmp_acc1_vec = tmp_acc1_vec + tmp11;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(131072L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp10 = tmp8 - tmp9;
                    auto tmp12 = static_cast<float>(7.62939453125e-06);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp16 = tmp15 * tmp15;
                    auto tmp17 = tmp14 * tmp16;
                    auto tmp18 = tmp10 * tmp17;
                    auto tmp19 = tmp7 - tmp18;
                    auto tmp21 = tmp20 * tmp13;
                    auto tmp22 = tmp19 - tmp21;
                    auto tmp24 = tmp15 * tmp23;
                    auto tmp25 = tmp22 * tmp24;
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_345, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, clone, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_17, (192, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_23, (640, ), (1, ))
    assert_size_stride(primals_25, (640, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_31, (640, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_35, (160, ), (1, ))
    assert_size_stride(primals_37, (640, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_43, (640, ), (1, ))
    assert_size_stride(primals_45, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_49, (640, ), (1, ))
    assert_size_stride(primals_51, (160, ), (1, ))
    assert_size_stride(primals_53, (160, ), (1, ))
    assert_size_stride(primals_55, (640, ), (1, ))
    assert_size_stride(primals_57, (1920, ), (1, ))
    assert_size_stride(primals_59, (1920, ), (1, ))
    assert_size_stride(primals_61, (640, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_65, (1920, ), (1, ))
    assert_size_stride(primals_67, (1920, ), (1, ))
    assert_size_stride(primals_69, (640, ), (1, ))
    assert_size_stride(primals_71, (1920, ), (1, ))
    assert_size_stride(primals_73, (1920, ), (1, ))
    assert_size_stride(primals_75, (640, ), (1, ))
    assert_size_stride(primals_77, (1920, ), (1, ))
    assert_size_stride(primals_79, (1920, ), (1, ))
    assert_size_stride(primals_81, (640, ), (1, ))
    assert_size_stride(primals_83, (1920, ), (1, ))
    assert_size_stride(primals_85, (1920, ), (1, ))
    assert_size_stride(primals_87, (640, ), (1, ))
    assert_size_stride(primals_89, (1920, ), (1, ))
    assert_size_stride(primals_91, (1920, ), (1, ))
    assert_size_stride(primals_93, (640, ), (1, ))
    assert_size_stride(primals_95, (1920, ), (1, ))
    assert_size_stride(primals_97, (1920, ), (1, ))
    assert_size_stride(primals_99, (640, ), (1, ))
    assert_size_stride(primals_101, (1920, ), (1, ))
    assert_size_stride(primals_103, (1920, ), (1, ))
    assert_size_stride(primals_105, (640, ), (1, ))
    assert_size_stride(primals_107, (1920, ), (1, ))
    assert_size_stride(primals_109, (1920, ), (1, ))
    assert_size_stride(primals_111, (640, ), (1, ))
    assert_size_stride(primals_113, (2560, ), (1, ))
    assert_size_stride(primals_115, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_116, (128, 32, 3, 3), (288, 1, 96, 32))
    assert_size_stride(primals_117, (128, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_118, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_119, (192, 128, 3, 3), (1152, 1, 384, 128))
    assert_size_stride(primals_120, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_121, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_122, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_123, (192, 192, 3, 3), (1728, 1, 576, 192))
    assert_size_stride(primals_124, (160, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_125, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_126, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_127, (640, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_128, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_129, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_130, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_131, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_132, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_133, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_134, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_135, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_136, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_137, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_138, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_139, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_140, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_141, (160, 160, 3, 3), (1440, 1, 480, 160))
    assert_size_stride(primals_142, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_144, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_145, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_146, (640, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_147, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_148, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_149, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_150, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_151, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_152, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_153, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_154, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_155, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_156, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_157, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_159, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_160, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_161, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_162, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_163, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_164, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_165, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_166, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_167, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_168, (1920, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_169, (1920, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (640, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_171, (2560, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_345, (8, 3, 256, 256), (196608, 1, 768, 3))
    assert_size_stride(convolution, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(squeeze_1, (32, ), (1, ))
    assert_size_stride(relu, (8, 32, 128, 128), (524288, 1, 4096, 32))
    assert_size_stride(convolution_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_4, (128, ), (1, ))
    assert_size_stride(relu_1, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_2, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_7, (128, ), (1, ))
    assert_size_stride(convolution_3, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(squeeze_10, (128, ), (1, ))
    assert_size_stride(relu_2, (8, 128, 64, 64), (524288, 1, 8192, 128))
    assert_size_stride(convolution_4, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_13, (192, ), (1, ))
    assert_size_stride(relu_3, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_5, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_16, (192, ), (1, ))
    assert_size_stride(convolution_6, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_19, (192, ), (1, ))
    assert_size_stride(relu_4, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_7, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_22, (192, ), (1, ))
    assert_size_stride(relu_5, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_8, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(squeeze_25, (192, ), (1, ))
    assert_size_stride(relu_6, (8, 192, 32, 32), (196608, 1, 6144, 192))
    assert_size_stride(convolution_9, (8, 160, 32, 32), (163840, 1, 5120, 160))
    assert_size_stride(squeeze_28, (160, ), (1, ))
    assert_size_stride(relu_7, (8, 160, 32, 32), (163840, 1, 5120, 160))
    assert_size_stride(convolution_10, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_31, (160, ), (1, ))
    assert_size_stride(relu_8, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_11, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_34, (640, ), (1, ))
    assert_size_stride(convolution_12, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_37, (640, ), (1, ))
    assert_size_stride(relu_9, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_13, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_40, (160, ), (1, ))
    assert_size_stride(relu_10, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_14, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_43, (160, ), (1, ))
    assert_size_stride(relu_11, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_15, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_46, (640, ), (1, ))
    assert_size_stride(relu_12, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_16, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_49, (160, ), (1, ))
    assert_size_stride(relu_13, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_17, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_52, (160, ), (1, ))
    assert_size_stride(relu_14, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_18, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_55, (640, ), (1, ))
    assert_size_stride(relu_15, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_19, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_58, (160, ), (1, ))
    assert_size_stride(relu_16, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_20, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_61, (160, ), (1, ))
    assert_size_stride(relu_17, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_21, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_64, (640, ), (1, ))
    assert_size_stride(relu_18, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_22, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_67, (160, ), (1, ))
    assert_size_stride(relu_19, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_23, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_70, (160, ), (1, ))
    assert_size_stride(relu_20, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_24, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_73, (640, ), (1, ))
    assert_size_stride(relu_21, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_25, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_76, (160, ), (1, ))
    assert_size_stride(relu_22, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_26, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(squeeze_79, (160, ), (1, ))
    assert_size_stride(relu_23, (8, 160, 16, 16), (40960, 1, 2560, 160))
    assert_size_stride(convolution_27, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(squeeze_82, (640, ), (1, ))
    assert_size_stride(relu_24, (8, 640, 16, 16), (163840, 1, 10240, 640))
    assert_size_stride(convolution_28, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    assert_size_stride(squeeze_85, (1920, ), (1, ))
    assert_size_stride(relu_25, (8, 1920, 16, 16), (491520, 1, 30720, 1920))
    assert_size_stride(convolution_29, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_88, (1920, ), (1, ))
    assert_size_stride(relu_26, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_30, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_91, (640, ), (1, ))
    assert_size_stride(convolution_31, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_94, (640, ), (1, ))
    assert_size_stride(relu_27, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_32, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_97, (1920, ), (1, ))
    assert_size_stride(relu_28, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_33, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_100, (1920, ), (1, ))
    assert_size_stride(relu_29, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_34, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_103, (640, ), (1, ))
    assert_size_stride(relu_30, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_35, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_106, (1920, ), (1, ))
    assert_size_stride(relu_31, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_36, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_109, (1920, ), (1, ))
    assert_size_stride(relu_32, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_37, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_112, (640, ), (1, ))
    assert_size_stride(relu_33, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_38, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_115, (1920, ), (1, ))
    assert_size_stride(relu_34, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_39, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_118, (1920, ), (1, ))
    assert_size_stride(relu_35, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_40, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_121, (640, ), (1, ))
    assert_size_stride(relu_36, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_41, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_124, (1920, ), (1, ))
    assert_size_stride(relu_37, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_42, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_127, (1920, ), (1, ))
    assert_size_stride(relu_38, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_43, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_130, (640, ), (1, ))
    assert_size_stride(relu_39, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_44, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_133, (1920, ), (1, ))
    assert_size_stride(relu_40, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_45, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_136, (1920, ), (1, ))
    assert_size_stride(relu_41, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_46, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_139, (640, ), (1, ))
    assert_size_stride(relu_42, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_47, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_142, (1920, ), (1, ))
    assert_size_stride(relu_43, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_48, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_145, (1920, ), (1, ))
    assert_size_stride(relu_44, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_49, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_148, (640, ), (1, ))
    assert_size_stride(relu_45, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_50, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_151, (1920, ), (1, ))
    assert_size_stride(relu_46, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_51, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_154, (1920, ), (1, ))
    assert_size_stride(relu_47, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_52, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_157, (640, ), (1, ))
    assert_size_stride(relu_48, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_53, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_160, (1920, ), (1, ))
    assert_size_stride(relu_49, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_54, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(squeeze_163, (1920, ), (1, ))
    assert_size_stride(relu_50, (8, 1920, 8, 8), (122880, 1, 15360, 1920))
    assert_size_stride(convolution_55, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(squeeze_166, (640, ), (1, ))
    assert_size_stride(relu_51, (8, 640, 8, 8), (40960, 1, 5120, 640))
    assert_size_stride(convolution_56, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    assert_size_stride(squeeze_169, (2560, ), (1, ))
    assert_size_stride(clone, (8, 2560), (2560, 1))
    assert_size_stride(permute_1, (1000, 2560), (2560, 1))
    assert_size_stride(le, (8, 2560, 8, 8), (163840, 1, 20480, 2560))
    assert_size_stride(unsqueeze_230, (1, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(unsqueeze_242, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_254, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_266, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_278, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_290, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_302, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_314, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_326, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_338, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_350, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_362, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_374, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_386, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_398, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_410, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_422, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_434, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_446, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_458, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_470, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_482, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_494, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_506, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_518, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_530, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_542, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_554, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_566, (1, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(unsqueeze_578, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_590, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_602, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_614, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_626, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_638, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_650, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_662, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_674, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_686, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_698, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_710, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_722, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_734, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_746, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_758, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_770, (1, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(unsqueeze_782, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_794, (1, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(unsqueeze_806, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_818, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_830, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_842, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_854, (1, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(unsqueeze_866, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_878, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_890, (1, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(unsqueeze_902, (1, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 2560), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2560), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf5 = empty((2560, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_230.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del buf4
    del convolution_56
    del le
    del primals_113
    del squeeze_169
    del tangents_1
    del unsqueeze_230
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, relu_51, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_171
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf10 = empty((640, ), device='cpu', dtype=torch.float32)
    buf11 = empty((640, ), device='cpu', dtype=torch.float32)
    buf12 = empty((640, ), device='cpu', dtype=torch.float32)
    buf13 = empty_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_1(c_void_p(relu_51.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_242.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()))
    del convolution_55
    del primals_111
    del squeeze_166
    del unsqueeze_242
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf14 = aten.convolution_backward(buf13, relu_50, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_170
    buf15 = buf14[0]
    buf16 = buf14[1]
    del buf14
    buf17 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf18 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf19 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf20 = buf15; del buf15  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_2(c_void_p(buf20.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_254.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf17.data_ptr()), c_void_p(buf18.data_ptr()), c_void_p(buf19.data_ptr()))
    del convolution_54
    del primals_109
    del relu_50
    del squeeze_163
    del unsqueeze_254
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf21 = aten.convolution_backward(buf20, relu_49, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf20
    del primals_169
    buf22 = buf21[0]
    buf23 = buf21[1]
    del buf21
    buf24 = buf18; del buf18  # reuse
    buf25 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf26 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf27 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_3(c_void_p(buf27.data_ptr()), c_void_p(relu_49.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_266.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(buf25.data_ptr()), c_void_p(buf26.data_ptr()))
    del convolution_53
    del primals_107
    del relu_49
    del squeeze_160
    del unsqueeze_266
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, relu_48, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf27
    del primals_168
    buf29 = buf28[0]
    buf30 = buf28[1]
    del buf28
    buf31 = buf11; del buf11  # reuse
    buf32 = empty((640, ), device='cpu', dtype=torch.float32)
    buf33 = buf13; del buf13  # reuse
    buf35 = buf33; del buf33  # reuse
    buf34 = buf32; del buf32  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_4(c_void_p(buf35.data_ptr()), c_void_p(buf34.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_278.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf31.data_ptr()))
    del convolution_52
    del primals_105
    del squeeze_157
    del unsqueeze_278
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf36 = aten.convolution_backward(buf35, relu_47, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_167
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = buf25; del buf25  # reuse
    buf40 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf41 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf42 = buf37; del buf37  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf42.data_ptr()), c_void_p(relu_47.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_290.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(buf40.data_ptr()), c_void_p(buf41.data_ptr()))
    del convolution_51
    del primals_103
    del relu_47
    del squeeze_154
    del unsqueeze_290
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf43 = aten.convolution_backward(buf42, relu_46, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf42
    del primals_166
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf46 = buf40; del buf40  # reuse
    buf47 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf48 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf49 = buf44; del buf44  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf49.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_302.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf47.data_ptr()), c_void_p(buf48.data_ptr()))
    del convolution_50
    del primals_101
    del relu_46
    del squeeze_151
    del unsqueeze_302
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf50 = aten.convolution_backward(buf49, relu_45, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf49
    del primals_165
    buf51 = buf50[0]
    buf52 = buf50[1]
    del buf50
    buf53 = buf29; del buf29  # reuse
    buf54 = empty((640, ), device='cpu', dtype=torch.float32)
    buf55 = empty((640, ), device='cpu', dtype=torch.float32)
    buf56 = empty((640, ), device='cpu', dtype=torch.float32)
    buf57 = buf35; del buf35  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_7(c_void_p(buf53.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(relu_51.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_314.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf54.data_ptr()), c_void_p(buf55.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(buf57.data_ptr()))
    del buf51
    del buf8
    del convolution_49
    del primals_99
    del relu_45
    del relu_48
    del relu_51
    del squeeze_148
    del unsqueeze_314
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf58 = aten.convolution_backward(buf57, relu_44, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_164
    buf59 = buf58[0]
    buf60 = buf58[1]
    del buf58
    buf61 = buf47; del buf47  # reuse
    buf62 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf63 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf64 = buf59; del buf59  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_8(c_void_p(buf64.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_326.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(buf63.data_ptr()))
    del convolution_48
    del primals_97
    del relu_44
    del squeeze_145
    del unsqueeze_326
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf65 = aten.convolution_backward(buf64, relu_43, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf64
    del primals_163
    buf66 = buf65[0]
    buf67 = buf65[1]
    del buf65
    buf68 = buf62; del buf62  # reuse
    buf69 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf70 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf71 = buf66; del buf66  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_9(c_void_p(buf71.data_ptr()), c_void_p(relu_43.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_338.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(buf70.data_ptr()))
    del convolution_47
    del primals_95
    del relu_43
    del squeeze_142
    del unsqueeze_338
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf72 = aten.convolution_backward(buf71, relu_42, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf71
    del primals_162
    buf73 = buf72[0]
    buf74 = buf72[1]
    del buf72
    buf75 = buf55; del buf55  # reuse
    buf76 = empty((640, ), device='cpu', dtype=torch.float32)
    buf77 = buf57; del buf57  # reuse
    buf78 = buf76; del buf76  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_10(c_void_p(buf78.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(buf53.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_350.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf77.data_ptr()))
    del convolution_46
    del primals_93
    del squeeze_139
    del unsqueeze_350
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf79 = aten.convolution_backward(buf77, relu_41, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_161
    buf80 = buf79[0]
    buf81 = buf79[1]
    del buf79
    buf82 = buf69; del buf69  # reuse
    buf83 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf84 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf85 = buf80; del buf80  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf85.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_362.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf82.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()))
    del convolution_45
    del primals_91
    del relu_41
    del squeeze_136
    del unsqueeze_362
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf86 = aten.convolution_backward(buf85, relu_40, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf85
    del primals_160
    buf87 = buf86[0]
    buf88 = buf86[1]
    del buf86
    buf89 = buf83; del buf83  # reuse
    buf90 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf91 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf92 = buf87; del buf87  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_12(c_void_p(buf92.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_374.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(buf90.data_ptr()), c_void_p(buf91.data_ptr()))
    del convolution_44
    del primals_89
    del relu_40
    del squeeze_133
    del unsqueeze_374
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf93 = aten.convolution_backward(buf92, relu_39, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf92
    del primals_159
    buf94 = buf93[0]
    buf95 = buf93[1]
    del buf93
    buf96 = buf53; del buf53  # reuse
    buf97 = empty((640, ), device='cpu', dtype=torch.float32)
    buf98 = empty((640, ), device='cpu', dtype=torch.float32)
    buf99 = empty((640, ), device='cpu', dtype=torch.float32)
    buf100 = buf77; del buf77  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_13(c_void_p(buf96.data_ptr()), c_void_p(relu_39.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf94.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_386.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()))
    del buf73
    del buf94
    del convolution_43
    del primals_87
    del relu_39
    del relu_42
    del squeeze_130
    del unsqueeze_386
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf101 = aten.convolution_backward(buf100, relu_38, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_158
    buf102 = buf101[0]
    buf103 = buf101[1]
    del buf101
    buf104 = buf90; del buf90  # reuse
    buf105 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf106 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf107 = buf102; del buf102  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_14(c_void_p(buf107.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_398.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()))
    del convolution_42
    del primals_85
    del relu_38
    del squeeze_127
    del unsqueeze_398
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf108 = aten.convolution_backward(buf107, relu_37, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf107
    del primals_157
    buf109 = buf108[0]
    buf110 = buf108[1]
    del buf108
    buf111 = buf105; del buf105  # reuse
    buf112 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf113 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf114 = buf109; del buf109  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(buf114.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_410.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(buf113.data_ptr()))
    del convolution_41
    del primals_83
    del relu_37
    del squeeze_124
    del unsqueeze_410
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf115 = aten.convolution_backward(buf114, relu_36, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf114
    del primals_156
    buf116 = buf115[0]
    buf117 = buf115[1]
    del buf115
    buf118 = buf98; del buf98  # reuse
    buf119 = empty((640, ), device='cpu', dtype=torch.float32)
    buf120 = buf100; del buf100  # reuse
    buf121 = buf119; del buf119  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_16(c_void_p(buf121.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_422.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf118.data_ptr()), c_void_p(buf120.data_ptr()))
    del convolution_40
    del primals_81
    del squeeze_121
    del unsqueeze_422
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf122 = aten.convolution_backward(buf120, relu_35, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_155
    buf123 = buf122[0]
    buf124 = buf122[1]
    del buf122
    buf125 = buf112; del buf112  # reuse
    buf126 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf127 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf128 = buf123; del buf123  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_17(c_void_p(buf128.data_ptr()), c_void_p(relu_35.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_434.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()))
    del convolution_39
    del primals_79
    del relu_35
    del squeeze_118
    del unsqueeze_434
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf129 = aten.convolution_backward(buf128, relu_34, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf128
    del primals_154
    buf130 = buf129[0]
    buf131 = buf129[1]
    del buf129
    buf132 = buf126; del buf126  # reuse
    buf133 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf134 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf135 = buf130; del buf130  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_18(c_void_p(buf135.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_446.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()))
    del convolution_38
    del primals_77
    del relu_34
    del squeeze_115
    del unsqueeze_446
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf136 = aten.convolution_backward(buf135, relu_33, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf135
    del primals_153
    buf137 = buf136[0]
    buf138 = buf136[1]
    del buf136
    buf139 = buf116; del buf116  # reuse
    buf140 = empty((640, ), device='cpu', dtype=torch.float32)
    buf141 = empty((640, ), device='cpu', dtype=torch.float32)
    buf142 = empty((640, ), device='cpu', dtype=torch.float32)
    buf143 = buf120; del buf120  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf139.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(buf137.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_458.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del buf137
    del convolution_37
    del primals_75
    del relu_33
    del relu_36
    del squeeze_112
    del unsqueeze_458
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf144 = aten.convolution_backward(buf143, relu_32, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_152
    buf145 = buf144[0]
    buf146 = buf144[1]
    del buf144
    buf147 = buf133; del buf133  # reuse
    buf148 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf149 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf150 = buf145; del buf145  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_20(c_void_p(buf150.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_470.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del convolution_36
    del primals_73
    del relu_32
    del squeeze_109
    del unsqueeze_470
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf151 = aten.convolution_backward(buf150, relu_31, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf150
    del primals_151
    buf152 = buf151[0]
    buf153 = buf151[1]
    del buf151
    buf154 = buf148; del buf148  # reuse
    buf155 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf156 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf157 = buf152; del buf152  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_21(c_void_p(buf157.data_ptr()), c_void_p(relu_31.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_482.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()))
    del convolution_35
    del primals_71
    del relu_31
    del squeeze_106
    del unsqueeze_482
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf158 = aten.convolution_backward(buf157, relu_30, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf157
    del primals_150
    buf159 = buf158[0]
    buf160 = buf158[1]
    del buf158
    buf161 = buf141; del buf141  # reuse
    buf162 = empty((640, ), device='cpu', dtype=torch.float32)
    buf163 = buf143; del buf143  # reuse
    buf164 = buf162; del buf162  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_22(c_void_p(buf164.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_494.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf161.data_ptr()), c_void_p(buf163.data_ptr()))
    del convolution_34
    del primals_69
    del squeeze_103
    del unsqueeze_494
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf165 = aten.convolution_backward(buf163, relu_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_149
    buf166 = buf165[0]
    buf167 = buf165[1]
    del buf165
    buf168 = buf155; del buf155  # reuse
    buf169 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf170 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf171 = buf166; del buf166  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf171.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_506.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf168.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()))
    del convolution_33
    del primals_67
    del relu_29
    del squeeze_100
    del unsqueeze_506
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf172 = aten.convolution_backward(buf171, relu_28, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf171
    del primals_148
    buf173 = buf172[0]
    buf174 = buf172[1]
    del buf172
    buf175 = buf169; del buf169  # reuse
    buf176 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf177 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf178 = buf173; del buf173  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf178.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_518.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf175.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(buf177.data_ptr()))
    del convolution_32
    del primals_65
    del relu_28
    del squeeze_97
    del unsqueeze_518
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf179 = aten.convolution_backward(buf178, relu_27, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf178
    del primals_147
    buf180 = buf179[0]
    buf181 = buf179[1]
    del buf179
    buf182 = buf139; del buf139  # reuse
    buf183 = empty((640, ), device='cpu', dtype=torch.float32)
    buf184 = empty((640, ), device='cpu', dtype=torch.float32)
    buf190 = empty((640, ), device='cpu', dtype=torch.float32)
    buf185 = empty((640, ), device='cpu', dtype=torch.float32)
    buf186 = buf163; del buf163  # reuse
    buf192 = buf96; del buf96  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_25(c_void_p(buf182.data_ptr()), c_void_p(relu_27.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(buf159.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_530.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_542.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf183.data_ptr()), c_void_p(buf184.data_ptr()), c_void_p(buf190.data_ptr()), c_void_p(buf185.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(buf192.data_ptr()))
    del buf159
    del buf180
    del buf182
    del convolution_30
    del convolution_31
    del primals_61
    del primals_63
    del relu_27
    del relu_30
    del squeeze_94
    del unsqueeze_530
    del unsqueeze_542
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf187 = aten.convolution_backward(buf186, relu_24, primals_146, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf186
    del primals_146
    buf188 = buf187[0]
    buf189 = buf187[1]
    del buf187
    buf191 = buf190; del buf190  # reuse
    cpp_fused_native_batch_norm_backward_26(c_void_p(buf191.data_ptr()), c_void_p(squeeze_91.data_ptr()))
    del squeeze_91
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf193 = aten.convolution_backward(buf192, relu_26, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf192
    del primals_145
    buf194 = buf193[0]
    buf195 = buf193[1]
    del buf193
    buf196 = buf176; del buf176  # reuse
    buf197 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf198 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf199 = buf194; del buf194  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_27(c_void_p(buf199.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_554.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()), c_void_p(buf198.data_ptr()))
    del convolution_29
    del primals_59
    del relu_26
    del squeeze_88
    del unsqueeze_554
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf200 = aten.convolution_backward(buf199, relu_25, primals_144, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False])
    del buf199
    del primals_144
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = buf197; del buf197  # reuse
    buf204 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf205 = empty((1920, ), device='cpu', dtype=torch.float32)
    buf206 = buf201; del buf201  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(buf206.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_566.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()))
    del buf204
    del convolution_28
    del primals_57
    del relu_25
    del squeeze_85
    del unsqueeze_566
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf207 = aten.convolution_backward(buf206, relu_24, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf206
    del primals_143
    buf208 = buf207[0]
    buf209 = buf207[1]
    del buf207
    buf210 = buf184; del buf184  # reuse
    buf211 = empty((640, ), device='cpu', dtype=torch.float32)
    buf212 = reinterpret_tensor(buf6, (8, 640, 16, 16), (163840, 1, 10240, 640), 0); del buf6  # reuse
    buf213 = buf211; del buf211  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_29(c_void_p(buf213.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_578.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf210.data_ptr()), c_void_p(buf212.data_ptr()))
    del convolution_27
    del primals_55
    del squeeze_82
    del unsqueeze_578
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf214 = aten.convolution_backward(buf212, relu_23, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_142
    buf215 = buf214[0]
    buf216 = buf214[1]
    del buf214
    buf217 = empty((160, ), device='cpu', dtype=torch.float32)
    buf218 = empty((160, ), device='cpu', dtype=torch.float32)
    buf219 = empty((160, ), device='cpu', dtype=torch.float32)
    buf220 = buf215; del buf215  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_30(c_void_p(buf220.data_ptr()), c_void_p(relu_23.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_590.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf218.data_ptr()), c_void_p(buf219.data_ptr()))
    del convolution_26
    del primals_53
    del relu_23
    del squeeze_79
    del unsqueeze_590
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf221 = aten.convolution_backward(buf220, relu_22, primals_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf220
    del primals_141
    buf222 = buf221[0]
    buf223 = buf221[1]
    del buf221
    buf224 = buf218; del buf218  # reuse
    buf225 = empty((160, ), device='cpu', dtype=torch.float32)
    buf226 = empty((160, ), device='cpu', dtype=torch.float32)
    buf227 = buf222; del buf222  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_31(c_void_p(buf227.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_602.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(buf225.data_ptr()), c_void_p(buf226.data_ptr()))
    del convolution_25
    del primals_51
    del relu_22
    del squeeze_76
    del unsqueeze_602
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf228 = aten.convolution_backward(buf227, relu_21, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf227
    del primals_140
    buf229 = buf228[0]
    buf230 = buf228[1]
    del buf228
    buf231 = buf188; del buf188  # reuse
    buf232 = empty((640, ), device='cpu', dtype=torch.float32)
    buf233 = empty((640, ), device='cpu', dtype=torch.float32)
    buf234 = empty((640, ), device='cpu', dtype=torch.float32)
    buf235 = buf212; del buf212  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf231.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(buf229.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_614.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf232.data_ptr()), c_void_p(buf233.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf235.data_ptr()))
    del buf208
    del buf229
    del convolution_24
    del primals_49
    del relu_21
    del relu_24
    del squeeze_73
    del unsqueeze_614
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf236 = aten.convolution_backward(buf235, relu_20, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_139
    buf237 = buf236[0]
    buf238 = buf236[1]
    del buf236
    buf239 = buf225; del buf225  # reuse
    buf240 = empty((160, ), device='cpu', dtype=torch.float32)
    buf241 = empty((160, ), device='cpu', dtype=torch.float32)
    buf242 = buf237; del buf237  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_33(c_void_p(buf242.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_626.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(buf241.data_ptr()))
    del convolution_23
    del primals_47
    del relu_20
    del squeeze_70
    del unsqueeze_626
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf243 = aten.convolution_backward(buf242, relu_19, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf242
    del primals_138
    buf244 = buf243[0]
    buf245 = buf243[1]
    del buf243
    buf246 = buf240; del buf240  # reuse
    buf247 = empty((160, ), device='cpu', dtype=torch.float32)
    buf248 = empty((160, ), device='cpu', dtype=torch.float32)
    buf249 = buf244; del buf244  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_34(c_void_p(buf249.data_ptr()), c_void_p(relu_19.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_638.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(buf248.data_ptr()))
    del convolution_22
    del primals_45
    del relu_19
    del squeeze_67
    del unsqueeze_638
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf250 = aten.convolution_backward(buf249, relu_18, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf249
    del primals_137
    buf251 = buf250[0]
    buf252 = buf250[1]
    del buf250
    buf253 = buf233; del buf233  # reuse
    buf254 = empty((640, ), device='cpu', dtype=torch.float32)
    buf255 = buf235; del buf235  # reuse
    buf256 = buf254; del buf254  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_35(c_void_p(buf256.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(buf231.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_650.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf255.data_ptr()))
    del convolution_21
    del primals_43
    del squeeze_64
    del unsqueeze_650
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf257 = aten.convolution_backward(buf255, relu_17, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_136
    buf258 = buf257[0]
    buf259 = buf257[1]
    del buf257
    buf260 = buf247; del buf247  # reuse
    buf261 = empty((160, ), device='cpu', dtype=torch.float32)
    buf262 = empty((160, ), device='cpu', dtype=torch.float32)
    buf263 = buf258; del buf258  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf263.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_662.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()), c_void_p(buf262.data_ptr()))
    del convolution_20
    del primals_41
    del relu_17
    del squeeze_61
    del unsqueeze_662
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf264 = aten.convolution_backward(buf263, relu_16, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf263
    del primals_135
    buf265 = buf264[0]
    buf266 = buf264[1]
    del buf264
    buf267 = buf261; del buf261  # reuse
    buf268 = empty((160, ), device='cpu', dtype=torch.float32)
    buf269 = empty((160, ), device='cpu', dtype=torch.float32)
    buf270 = buf265; del buf265  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf270.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_674.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()))
    del convolution_19
    del primals_39
    del relu_16
    del squeeze_58
    del unsqueeze_674
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf271 = aten.convolution_backward(buf270, relu_15, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf270
    del primals_134
    buf272 = buf271[0]
    buf273 = buf271[1]
    del buf271
    buf274 = buf231; del buf231  # reuse
    buf275 = empty((640, ), device='cpu', dtype=torch.float32)
    buf276 = empty((640, ), device='cpu', dtype=torch.float32)
    buf277 = empty((640, ), device='cpu', dtype=torch.float32)
    buf278 = buf255; del buf255  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_38(c_void_p(buf274.data_ptr()), c_void_p(relu_15.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_686.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()), c_void_p(buf278.data_ptr()))
    del buf251
    del convolution_18
    del primals_37
    del relu_15
    del relu_18
    del squeeze_55
    del unsqueeze_686
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf279 = aten.convolution_backward(buf278, relu_14, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_133
    buf280 = buf279[0]
    buf281 = buf279[1]
    del buf279
    buf282 = buf268; del buf268  # reuse
    buf283 = empty((160, ), device='cpu', dtype=torch.float32)
    buf284 = empty((160, ), device='cpu', dtype=torch.float32)
    buf285 = buf280; del buf280  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_39(c_void_p(buf285.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_698.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf282.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()))
    del convolution_17
    del primals_35
    del relu_14
    del squeeze_52
    del unsqueeze_698
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf286 = aten.convolution_backward(buf285, relu_13, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf285
    del primals_132
    buf287 = buf286[0]
    buf288 = buf286[1]
    del buf286
    buf289 = buf283; del buf283  # reuse
    buf290 = empty((160, ), device='cpu', dtype=torch.float32)
    buf291 = empty((160, ), device='cpu', dtype=torch.float32)
    buf292 = buf287; del buf287  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_40(c_void_p(buf292.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_710.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf289.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf291.data_ptr()))
    del convolution_16
    del primals_33
    del relu_13
    del squeeze_49
    del unsqueeze_710
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf293 = aten.convolution_backward(buf292, relu_12, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf292
    del primals_131
    buf294 = buf293[0]
    buf295 = buf293[1]
    del buf293
    buf296 = buf276; del buf276  # reuse
    buf297 = empty((640, ), device='cpu', dtype=torch.float32)
    buf298 = buf278; del buf278  # reuse
    buf299 = buf297; del buf297  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_41(c_void_p(buf299.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_722.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf296.data_ptr()), c_void_p(buf298.data_ptr()))
    del convolution_15
    del primals_31
    del squeeze_46
    del unsqueeze_722
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf300 = aten.convolution_backward(buf298, relu_11, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_130
    buf301 = buf300[0]
    buf302 = buf300[1]
    del buf300
    buf303 = buf290; del buf290  # reuse
    buf304 = empty((160, ), device='cpu', dtype=torch.float32)
    buf305 = empty((160, ), device='cpu', dtype=torch.float32)
    buf306 = buf301; del buf301  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_42(c_void_p(buf306.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_734.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf303.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(buf305.data_ptr()))
    del convolution_14
    del primals_29
    del relu_11
    del squeeze_43
    del unsqueeze_734
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf307 = aten.convolution_backward(buf306, relu_10, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf306
    del primals_129
    buf308 = buf307[0]
    buf309 = buf307[1]
    del buf307
    buf310 = buf304; del buf304  # reuse
    buf311 = empty((160, ), device='cpu', dtype=torch.float32)
    buf312 = empty((160, ), device='cpu', dtype=torch.float32)
    buf313 = buf308; del buf308  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_43(c_void_p(buf313.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_746.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf310.data_ptr()), c_void_p(buf311.data_ptr()), c_void_p(buf312.data_ptr()))
    del convolution_13
    del primals_27
    del relu_10
    del squeeze_40
    del unsqueeze_746
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf314 = aten.convolution_backward(buf313, relu_9, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf313
    del primals_128
    buf315 = buf314[0]
    buf316 = buf314[1]
    del buf314
    buf317 = buf274; del buf274  # reuse
    buf318 = empty((640, ), device='cpu', dtype=torch.float32)
    buf319 = empty((640, ), device='cpu', dtype=torch.float32)
    buf325 = empty((640, ), device='cpu', dtype=torch.float32)
    buf320 = empty((640, ), device='cpu', dtype=torch.float32)
    buf321 = buf298; del buf298  # reuse
    buf327 = buf272; del buf272  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_44(c_void_p(buf317.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf294.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_758.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_770.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf318.data_ptr()), c_void_p(buf319.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(buf321.data_ptr()), c_void_p(buf327.data_ptr()))
    del buf294
    del buf315
    del buf317
    del buf319
    del convolution_11
    del convolution_12
    del primals_23
    del primals_25
    del relu_12
    del relu_9
    del squeeze_37
    del unsqueeze_758
    del unsqueeze_770
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf322 = aten.convolution_backward(buf321, relu_6, primals_127, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf321
    del primals_127
    buf323 = buf322[0]
    buf324 = buf322[1]
    del buf322
    buf326 = buf325; del buf325  # reuse
    cpp_fused_native_batch_norm_backward_45(c_void_p(buf326.data_ptr()), c_void_p(squeeze_34.data_ptr()))
    del squeeze_34
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf328 = aten.convolution_backward(buf327, relu_8, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf327
    del primals_126
    buf329 = buf328[0]
    buf330 = buf328[1]
    del buf328
    buf331 = buf311; del buf311  # reuse
    buf332 = empty((160, ), device='cpu', dtype=torch.float32)
    buf333 = empty((160, ), device='cpu', dtype=torch.float32)
    buf334 = buf329; del buf329  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_46(c_void_p(buf334.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_782.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()))
    del convolution_10
    del primals_21
    del relu_8
    del squeeze_31
    del unsqueeze_782
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf335 = aten.convolution_backward(buf334, relu_7, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf334
    del primals_125
    buf336 = buf335[0]
    buf337 = buf335[1]
    del buf335
    buf338 = buf332; del buf332  # reuse
    buf339 = empty((160, ), device='cpu', dtype=torch.float32)
    buf340 = empty((160, ), device='cpu', dtype=torch.float32)
    buf341 = buf336; del buf336  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_47(c_void_p(buf341.data_ptr()), c_void_p(relu_7.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_794.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf338.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()))
    del buf339
    del convolution_9
    del primals_19
    del relu_7
    del squeeze_28
    del unsqueeze_794
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf342 = aten.convolution_backward(buf341, relu_6, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf341
    del primals_124
    buf343 = buf342[0]
    buf344 = buf342[1]
    del buf342
    buf345 = empty((192, ), device='cpu', dtype=torch.float32)
    buf346 = empty((192, ), device='cpu', dtype=torch.float32)
    buf347 = empty_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    buf348 = buf346; del buf346  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_48(c_void_p(buf348.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_806.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf345.data_ptr()), c_void_p(buf347.data_ptr()))
    del convolution_8
    del primals_17
    del squeeze_25
    del unsqueeze_806
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf349 = aten.convolution_backward(buf347, relu_5, primals_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_123
    buf350 = buf349[0]
    buf351 = buf349[1]
    del buf349
    buf352 = empty((192, ), device='cpu', dtype=torch.float32)
    buf353 = empty((192, ), device='cpu', dtype=torch.float32)
    buf354 = empty((192, ), device='cpu', dtype=torch.float32)
    buf355 = buf350; del buf350  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf355.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_818.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(buf353.data_ptr()), c_void_p(buf354.data_ptr()))
    del convolution_7
    del primals_15
    del relu_5
    del squeeze_22
    del unsqueeze_818
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf356 = aten.convolution_backward(buf355, relu_4, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_122
    buf357 = buf356[0]
    buf358 = buf356[1]
    del buf356
    buf359 = buf323; del buf323  # reuse
    buf360 = buf353; del buf353  # reuse
    buf361 = empty((192, ), device='cpu', dtype=torch.float32)
    buf367 = empty((192, ), device='cpu', dtype=torch.float32)
    buf362 = empty((192, ), device='cpu', dtype=torch.float32)
    buf363 = buf355; del buf355  # reuse
    buf369 = buf347; del buf347  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf359.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_830.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_842.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf360.data_ptr()), c_void_p(buf361.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf362.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf369.data_ptr()))
    del buf343
    del buf357
    del buf359
    del convolution_5
    del convolution_6
    del primals_11
    del primals_13
    del relu_4
    del relu_6
    del squeeze_19
    del unsqueeze_830
    del unsqueeze_842
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf364 = aten.convolution_backward(buf363, relu_2, primals_121, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf363
    del primals_121
    buf365 = buf364[0]
    buf366 = buf364[1]
    del buf364
    buf368 = buf367; del buf367  # reuse
    cpp_fused_native_batch_norm_backward_51(c_void_p(buf368.data_ptr()), c_void_p(squeeze_16.data_ptr()))
    del squeeze_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf370 = aten.convolution_backward(buf369, relu_3, primals_120, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf369
    del primals_120
    buf371 = buf370[0]
    buf372 = buf370[1]
    del buf370
    buf373 = buf361; del buf361  # reuse
    buf374 = empty((192, ), device='cpu', dtype=torch.float32)
    buf375 = empty((192, ), device='cpu', dtype=torch.float32)
    buf376 = buf371; del buf371  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_52(c_void_p(buf376.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_854.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(buf374.data_ptr()), c_void_p(buf375.data_ptr()))
    del buf374
    del convolution_4
    del primals_9
    del relu_3
    del squeeze_13
    del unsqueeze_854
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf377 = aten.convolution_backward(buf376, relu_2, primals_119, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf376
    del primals_119
    buf378 = buf377[0]
    buf379 = buf377[1]
    del buf377
    buf380 = empty((128, ), device='cpu', dtype=torch.float32)
    buf381 = empty((128, ), device='cpu', dtype=torch.float32)
    buf387 = empty((128, ), device='cpu', dtype=torch.float32)
    buf382 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    buf388 = empty_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    buf383 = buf381; del buf381  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_53(c_void_p(buf383.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_866.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_878.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf382.data_ptr()), c_void_p(buf388.data_ptr()))
    del buf365
    del buf378
    del convolution_2
    del convolution_3
    del primals_5
    del primals_7
    del relu_2
    del squeeze_10
    del unsqueeze_866
    del unsqueeze_878
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf384 = aten.convolution_backward(buf382, relu, primals_118, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf382
    del primals_118
    buf385 = buf384[0]
    buf386 = buf384[1]
    del buf384
    buf389 = buf387; del buf387  # reuse
    cpp_fused_native_batch_norm_backward_54(c_void_p(buf389.data_ptr()), c_void_p(squeeze_7.data_ptr()))
    del squeeze_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf390 = aten.convolution_backward(buf388, relu_1, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf388
    del primals_117
    buf391 = buf390[0]
    buf392 = buf390[1]
    del buf390
    buf393 = empty((128, ), device='cpu', dtype=torch.float32)
    buf394 = empty((128, ), device='cpu', dtype=torch.float32)
    buf395 = empty((128, ), device='cpu', dtype=torch.float32)
    buf396 = buf391; del buf391  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_55(c_void_p(buf396.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_890.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(buf395.data_ptr()))
    del buf394
    del convolution_1
    del primals_3
    del relu_1
    del squeeze_4
    del unsqueeze_890
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf397 = aten.convolution_backward(buf396, relu, primals_116, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf396
    del primals_116
    buf398 = buf397[0]
    buf399 = buf397[1]
    del buf397
    buf400 = empty((32, ), device='cpu', dtype=torch.float32)
    buf401 = empty((32, ), device='cpu', dtype=torch.float32)
    buf402 = buf385; del buf385  # reuse
    buf403 = buf401; del buf401  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_56(c_void_p(buf402.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf398.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_902.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf400.data_ptr()))
    del buf398
    del convolution
    del primals_1
    del relu
    del squeeze_1
    del unsqueeze_902
    # Source Nodes: [], Original ATen: [aten.convolution_backward]
    buf404 = aten.convolution_backward(buf402, primals_345, primals_115, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf402
    del primals_115
    del primals_345
    buf405 = buf404[1]
    return (buf403, buf400, buf395, buf393, buf389, buf380, buf383, buf380, buf375, buf373, buf368, buf360, buf362, buf360, buf354, buf352, buf348, buf345, buf340, buf338, buf333, buf331, buf326, buf318, buf320, buf318, buf312, buf310, buf305, buf303, buf299, buf296, buf291, buf289, buf284, buf282, buf277, buf275, buf269, buf267, buf262, buf260, buf256, buf253, buf248, buf246, buf241, buf239, buf234, buf232, buf226, buf224, buf219, buf217, buf213, buf210, buf205, buf203, buf198, buf196, buf191, buf183, buf185, buf183, buf177, buf175, buf170, buf168, buf164, buf161, buf156, buf154, buf149, buf147, buf142, buf140, buf134, buf132, buf127, buf125, buf121, buf118, buf113, buf111, buf106, buf104, buf99, buf97, buf91, buf89, buf84, buf82, buf78, buf75, buf70, buf68, buf63, buf61, buf56, buf54, buf48, buf46, buf41, buf39, buf34, buf31, buf26, buf24, buf19, buf17, buf12, buf10, buf5, buf3, buf405, buf399, buf392, buf386, buf379, buf372, buf366, buf358, buf351, buf344, buf337, buf330, buf324, buf316, buf309, buf302, buf295, buf288, buf281, buf273, buf266, buf259, buf252, buf245, buf238, buf230, buf223, buf216, buf209, buf202, buf195, buf189, buf181, buf174, buf167, buf160, buf153, buf146, buf138, buf131, buf124, buf117, buf110, buf103, buf95, buf88, buf81, buf74, buf67, buf60, buf52, buf45, buf38, buf30, buf23, buf16, buf9, reinterpret_tensor(buf1, (1000, 2560), (2560, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_118 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((160, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((640, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((160, 160, 3, 3), (1440, 1, 480, 160), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_144 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((640, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_154 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((1920, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((1920, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((640, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((2560, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((8, 3, 256, 256), (196608, 1, 768, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    relu = rand_strided((8, 32, 128, 128), (524288, 1, 4096, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((128, ), (1, ), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((8, 128, 64, 64), (524288, 1, 8192, 128), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((192, ), (1, ), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((8, 192, 32, 32), (196608, 1, 6144, 192), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((8, 160, 32, 32), (163840, 1, 5120, 160), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((8, 160, 16, 16), (40960, 1, 2560, 160), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((8, 640, 16, 16), (163840, 1, 10240, 640), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((8, 1920, 16, 16), (491520, 1, 30720, 1920), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_48 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_49 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((1920, ), (1, ), device='cpu', dtype=torch.float32)
    relu_50 = rand_strided((8, 1920, 8, 8), (122880, 1, 15360, 1920), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((640, ), (1, ), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((8, 640, 8, 8), (40960, 1, 5120, 640), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((2560, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2560), (2560, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((8, 2560, 8, 8), (163840, 1, 20480, 2560), device='cpu', dtype=torch.bool)
    unsqueeze_230 = rand_strided((1, 2560, 1, 1), (2560, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_242 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_254 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_266 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_278 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_290 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_302 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_314 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_326 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_338 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_350 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_362 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_374 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_386 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_398 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_410 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_422 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_434 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_446 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_458 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_470 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_482 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_494 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_506 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_518 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_530 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_542 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_554 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_566 = rand_strided((1, 1920, 1, 1), (1920, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_578 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_590 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_602 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_614 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_626 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_638 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_650 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_662 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_674 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_686 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_698 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_710 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_722 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_734 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_746 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_758 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_770 = rand_strided((1, 640, 1, 1), (640, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_782 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_794 = rand_strided((1, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_806 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_818 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_830 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_842 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_854 = rand_strided((1, 192, 1, 1), (192, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_866 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_878 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_890 = rand_strided((1, 128, 1, 1), (128, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_902 = rand_strided((1, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_345, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, squeeze_7, convolution_3, squeeze_10, relu_2, convolution_4, squeeze_13, relu_3, convolution_5, squeeze_16, convolution_6, squeeze_19, relu_4, convolution_7, squeeze_22, relu_5, convolution_8, squeeze_25, relu_6, convolution_9, squeeze_28, relu_7, convolution_10, squeeze_31, relu_8, convolution_11, squeeze_34, convolution_12, squeeze_37, relu_9, convolution_13, squeeze_40, relu_10, convolution_14, squeeze_43, relu_11, convolution_15, squeeze_46, relu_12, convolution_16, squeeze_49, relu_13, convolution_17, squeeze_52, relu_14, convolution_18, squeeze_55, relu_15, convolution_19, squeeze_58, relu_16, convolution_20, squeeze_61, relu_17, convolution_21, squeeze_64, relu_18, convolution_22, squeeze_67, relu_19, convolution_23, squeeze_70, relu_20, convolution_24, squeeze_73, relu_21, convolution_25, squeeze_76, relu_22, convolution_26, squeeze_79, relu_23, convolution_27, squeeze_82, relu_24, convolution_28, squeeze_85, relu_25, convolution_29, squeeze_88, relu_26, convolution_30, squeeze_91, convolution_31, squeeze_94, relu_27, convolution_32, squeeze_97, relu_28, convolution_33, squeeze_100, relu_29, convolution_34, squeeze_103, relu_30, convolution_35, squeeze_106, relu_31, convolution_36, squeeze_109, relu_32, convolution_37, squeeze_112, relu_33, convolution_38, squeeze_115, relu_34, convolution_39, squeeze_118, relu_35, convolution_40, squeeze_121, relu_36, convolution_41, squeeze_124, relu_37, convolution_42, squeeze_127, relu_38, convolution_43, squeeze_130, relu_39, convolution_44, squeeze_133, relu_40, convolution_45, squeeze_136, relu_41, convolution_46, squeeze_139, relu_42, convolution_47, squeeze_142, relu_43, convolution_48, squeeze_145, relu_44, convolution_49, squeeze_148, relu_45, convolution_50, squeeze_151, relu_46, convolution_51, squeeze_154, relu_47, convolution_52, squeeze_157, relu_48, convolution_53, squeeze_160, relu_49, convolution_54, squeeze_163, relu_50, convolution_55, squeeze_166, relu_51, convolution_56, squeeze_169, clone, permute_1, le, unsqueeze_230, unsqueeze_242, unsqueeze_254, unsqueeze_266, unsqueeze_278, unsqueeze_290, unsqueeze_302, unsqueeze_314, unsqueeze_326, unsqueeze_338, unsqueeze_350, unsqueeze_362, unsqueeze_374, unsqueeze_386, unsqueeze_398, unsqueeze_410, unsqueeze_422, unsqueeze_434, unsqueeze_446, unsqueeze_458, unsqueeze_470, unsqueeze_482, unsqueeze_494, unsqueeze_506, unsqueeze_518, unsqueeze_530, unsqueeze_542, unsqueeze_554, unsqueeze_566, unsqueeze_578, unsqueeze_590, unsqueeze_602, unsqueeze_614, unsqueeze_626, unsqueeze_638, unsqueeze_650, unsqueeze_662, unsqueeze_674, unsqueeze_686, unsqueeze_698, unsqueeze_710, unsqueeze_722, unsqueeze_734, unsqueeze_746, unsqueeze_758, unsqueeze_770, unsqueeze_782, unsqueeze_794, unsqueeze_806, unsqueeze_818, unsqueeze_830, unsqueeze_842, unsqueeze_854, unsqueeze_866, unsqueeze_878, unsqueeze_890, unsqueeze_902, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gernet_l', benchmark_compiled_module)
