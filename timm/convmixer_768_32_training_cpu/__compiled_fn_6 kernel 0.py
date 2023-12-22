
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_relu_sum_threshold_backward_0 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(1024L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x2) + (786432L*x1)));
                            auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(1024.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = tmp0 / tmp2;
                            auto tmp5 = at::vec::clamp_min(tmp4, decltype(tmp4)(0));
                            auto tmp7 = tmp5 - tmp6;
                            auto tmp8 = tmp3 * tmp7;
                            tmp_acc0_vec = tmp_acc0_vec + tmp3;
                            tmp_acc1_vec = tmp_acc1_vec + tmp8;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(1024L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(768L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (768L*x1) + (786432L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (768L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2));
                        auto tmp11 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x2));
                        auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x2));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                        auto tmp2 = static_cast<float>(0.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = to_float_mask(tmp1 <= tmp3);
                        auto tmp6 = static_cast<float>(1024.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = tmp5 / tmp7;
                        auto tmp10 = tmp1 - tmp9;
                        auto tmp12 = static_cast<float>(0.0001220703125);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 * tmp13;
                        auto tmp16 = tmp15 * tmp15;
                        auto tmp17 = tmp14 * tmp16;
                        auto tmp18 = tmp10 * tmp17;
                        auto tmp19 = tmp8 - tmp18;
                        auto tmp21 = tmp20 * tmp13;
                        auto tmp22 = tmp19 - tmp21;
                        auto tmp24 = tmp15 * tmp23;
                        auto tmp25 = tmp22 * tmp24;
                        auto tmp26 = decltype(tmp3)::blendv(tmp25, tmp3, tmp4);
                        tmp26.store(out_ptr4 + static_cast<long>(x2 + (768L*x1) + (786432L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_2 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_11 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_14 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_18 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_19 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_24 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_27 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_31 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_32 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_37 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_40 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_42 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_43 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_44 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_50 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_53 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_57 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_63 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp2 = at::vec::clamp_min(tmp1, decltype(tmp1)(0));
                        auto tmp4 = tmp2 - tmp3;
                        auto tmp5 = tmp0 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp5;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = tmp0 * tmp1;
                    tmp2.store(out_ptr2 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(8192L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp1 - tmp6;
                    auto tmp9 = static_cast<float>(0.0001220703125);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 * tmp10;
                    auto tmp13 = tmp12 * tmp12;
                    auto tmp14 = tmp11 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp16 = tmp5 - tmp15;
                    auto tmp18 = tmp17 * tmp10;
                    auto tmp19 = tmp16 - tmp18;
                    auto tmp21 = tmp12 * tmp20;
                    auto tmp22 = tmp19 * tmp21;
                    auto tmp23 = decltype(tmp3)::blendv(tmp22, tmp3, tmp4);
                    tmp23.store(out_ptr3 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2)
{
    auto in_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(8192L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (768L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = at::vec::clamp_min(tmp3, decltype(tmp3)(0));
                        auto tmp6 = tmp4 - tmp5;
                        auto tmp7 = tmp2 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp7;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x1));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = at::vec::clamp_min(tmp0, decltype(tmp0)(0));
                    auto tmp2 = static_cast<float>(0.0);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = to_float_mask(tmp1 <= tmp3);
                    auto tmp7 = tmp5 + tmp6;
                    auto tmp9 = tmp1 - tmp8;
                    auto tmp11 = static_cast<float>(0.0001220703125);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 * tmp12;
                    auto tmp15 = tmp14 * tmp14;
                    auto tmp16 = tmp13 * tmp15;
                    auto tmp17 = tmp9 * tmp16;
                    auto tmp18 = tmp7 - tmp17;
                    auto tmp20 = tmp19 * tmp12;
                    auto tmp21 = tmp18 - tmp20;
                    auto tmp23 = tmp14 * tmp22;
                    auto tmp24 = tmp21 * tmp23;
                    auto tmp25 = decltype(tmp3)::blendv(tmp24, tmp3, tmp4);
                    tmp25.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_458, convolution, squeeze_1, add_4, convolution_1, squeeze_4, add_10, convolution_2, squeeze_7, add_15, convolution_3, squeeze_10, add_21, convolution_4, squeeze_13, add_26, convolution_5, squeeze_16, add_32, convolution_6, squeeze_19, add_37, convolution_7, squeeze_22, add_43, convolution_8, squeeze_25, add_48, convolution_9, squeeze_28, add_54, convolution_10, squeeze_31, add_59, convolution_11, squeeze_34, add_65, convolution_12, squeeze_37, add_70, convolution_13, squeeze_40, add_76, convolution_14, squeeze_43, add_81, convolution_15, squeeze_46, add_87, convolution_16, squeeze_49, add_92, convolution_17, squeeze_52, add_98, convolution_18, squeeze_55, add_103, convolution_19, squeeze_58, add_109, convolution_20, squeeze_61, add_114, convolution_21, squeeze_64, add_120, convolution_22, squeeze_67, add_125, convolution_23, squeeze_70, add_131, convolution_24, squeeze_73, add_136, convolution_25, squeeze_76, add_142, convolution_26, squeeze_79, add_147, convolution_27, squeeze_82, add_153, convolution_28, squeeze_85, add_158, convolution_29, squeeze_88, add_164, convolution_30, squeeze_91, add_169, convolution_31, squeeze_94, add_175, convolution_32, squeeze_97, add_180, convolution_33, squeeze_100, add_186, convolution_34, squeeze_103, add_191, convolution_35, squeeze_106, add_197, convolution_36, squeeze_109, add_202, convolution_37, squeeze_112, add_208, convolution_38, squeeze_115, add_213, convolution_39, squeeze_118, add_219, convolution_40, squeeze_121, add_224, convolution_41, squeeze_124, add_230, convolution_42, squeeze_127, add_235, convolution_43, squeeze_130, add_241, convolution_44, squeeze_133, add_246, convolution_45, squeeze_136, add_252, convolution_46, squeeze_139, add_257, convolution_47, squeeze_142, add_263, convolution_48, squeeze_145, add_268, convolution_49, squeeze_148, add_274, convolution_50, squeeze_151, add_279, convolution_51, squeeze_154, add_285, convolution_52, squeeze_157, add_290, convolution_53, squeeze_160, add_296, convolution_54, squeeze_163, add_301, convolution_55, squeeze_166, add_307, convolution_56, squeeze_169, add_312, convolution_57, squeeze_172, add_318, convolution_58, squeeze_175, add_323, convolution_59, squeeze_178, add_329, convolution_60, squeeze_181, add_334, convolution_61, squeeze_184, add_340, convolution_62, squeeze_187, add_345, convolution_63, squeeze_190, add_351, convolution_64, squeeze_193, clone, permute_1, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (768, 3, 7, 7), (147, 1, 21, 3))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_5, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_9, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_13, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_15, (768, ), (1, ))
    assert_size_stride(primals_17, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_19, (768, ), (1, ))
    assert_size_stride(primals_21, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_25, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_29, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_31, (768, ), (1, ))
    assert_size_stride(primals_33, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_35, (768, ), (1, ))
    assert_size_stride(primals_37, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_41, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_45, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_47, (768, ), (1, ))
    assert_size_stride(primals_49, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_53, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_57, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_59, (768, ), (1, ))
    assert_size_stride(primals_61, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_65, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_69, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_73, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_75, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_81, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_85, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_89, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_91, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_97, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_101, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_103, (768, ), (1, ))
    assert_size_stride(primals_105, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_109, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_113, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_117, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_121, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_125, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_127, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_133, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_137, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    assert_size_stride(primals_141, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_143, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_149, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_151, (768, ), (1, ))
    assert_size_stride(primals_153, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_157, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_159, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_165, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_169, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_171, (768, ), (1, ))
    assert_size_stride(primals_173, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_175, (768, ), (1, ))
    assert_size_stride(primals_177, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_181, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_183, (768, ), (1, ))
    assert_size_stride(primals_185, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_187, (768, ), (1, ))
    assert_size_stride(primals_189, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_193, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_195, (768, ), (1, ))
    assert_size_stride(primals_197, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_201, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_205, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_209, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_211, (768, ), (1, ))
    assert_size_stride(primals_213, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_217, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_221, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_225, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_229, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_233, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_237, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_241, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_243, (768, ), (1, ))
    assert_size_stride(primals_245, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_253, (768, 1, 7, 7), (49, 49, 7, 1))
    assert_size_stride(primals_255, (768, ), (1, ))
    assert_size_stride(primals_257, (768, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (768, ), (1, ))
    assert_size_stride(primals_458, (8, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_1, (768, ), (1, ))
    assert_size_stride(add_4, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_1, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_4, (768, ), (1, ))
    assert_size_stride(add_10, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_2, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_7, (768, ), (1, ))
    assert_size_stride(add_15, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_3, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_10, (768, ), (1, ))
    assert_size_stride(add_21, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_4, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_13, (768, ), (1, ))
    assert_size_stride(add_26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_5, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_16, (768, ), (1, ))
    assert_size_stride(add_32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_6, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_19, (768, ), (1, ))
    assert_size_stride(add_37, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_7, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_22, (768, ), (1, ))
    assert_size_stride(add_43, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_8, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_25, (768, ), (1, ))
    assert_size_stride(add_48, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_9, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_28, (768, ), (1, ))
    assert_size_stride(add_54, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_10, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_31, (768, ), (1, ))
    assert_size_stride(add_59, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_11, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_34, (768, ), (1, ))
    assert_size_stride(add_65, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_12, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_37, (768, ), (1, ))
    assert_size_stride(add_70, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_13, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_40, (768, ), (1, ))
    assert_size_stride(add_76, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_14, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_43, (768, ), (1, ))
    assert_size_stride(add_81, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_15, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_46, (768, ), (1, ))
    assert_size_stride(add_87, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_16, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_49, (768, ), (1, ))
    assert_size_stride(add_92, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_17, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_52, (768, ), (1, ))
    assert_size_stride(add_98, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_18, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_55, (768, ), (1, ))
    assert_size_stride(add_103, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_19, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_58, (768, ), (1, ))
    assert_size_stride(add_109, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_20, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_61, (768, ), (1, ))
    assert_size_stride(add_114, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_21, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_64, (768, ), (1, ))
    assert_size_stride(add_120, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_22, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_67, (768, ), (1, ))
    assert_size_stride(add_125, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_23, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_70, (768, ), (1, ))
    assert_size_stride(add_131, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_24, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_73, (768, ), (1, ))
    assert_size_stride(add_136, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_25, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_76, (768, ), (1, ))
    assert_size_stride(add_142, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_26, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_79, (768, ), (1, ))
    assert_size_stride(add_147, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_27, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_82, (768, ), (1, ))
    assert_size_stride(add_153, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_28, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_85, (768, ), (1, ))
    assert_size_stride(add_158, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_29, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_88, (768, ), (1, ))
    assert_size_stride(add_164, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_30, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_91, (768, ), (1, ))
    assert_size_stride(add_169, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_31, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_94, (768, ), (1, ))
    assert_size_stride(add_175, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_32, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_97, (768, ), (1, ))
    assert_size_stride(add_180, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_33, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_100, (768, ), (1, ))
    assert_size_stride(add_186, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_34, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_103, (768, ), (1, ))
    assert_size_stride(add_191, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_35, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_106, (768, ), (1, ))
    assert_size_stride(add_197, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_36, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_109, (768, ), (1, ))
    assert_size_stride(add_202, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_37, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_112, (768, ), (1, ))
    assert_size_stride(add_208, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_38, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_115, (768, ), (1, ))
    assert_size_stride(add_213, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_39, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_118, (768, ), (1, ))
    assert_size_stride(add_219, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_40, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_121, (768, ), (1, ))
    assert_size_stride(add_224, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_41, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_124, (768, ), (1, ))
    assert_size_stride(add_230, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_42, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_127, (768, ), (1, ))
    assert_size_stride(add_235, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_43, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_130, (768, ), (1, ))
    assert_size_stride(add_241, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_44, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_133, (768, ), (1, ))
    assert_size_stride(add_246, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_45, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_136, (768, ), (1, ))
    assert_size_stride(add_252, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_46, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_139, (768, ), (1, ))
    assert_size_stride(add_257, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_47, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_142, (768, ), (1, ))
    assert_size_stride(add_263, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_48, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_145, (768, ), (1, ))
    assert_size_stride(add_268, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_49, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_148, (768, ), (1, ))
    assert_size_stride(add_274, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_50, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_151, (768, ), (1, ))
    assert_size_stride(add_279, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_51, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_154, (768, ), (1, ))
    assert_size_stride(add_285, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_52, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_157, (768, ), (1, ))
    assert_size_stride(add_290, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_53, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_160, (768, ), (1, ))
    assert_size_stride(add_296, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_54, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_163, (768, ), (1, ))
    assert_size_stride(add_301, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_55, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_166, (768, ), (1, ))
    assert_size_stride(add_307, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_56, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_169, (768, ), (1, ))
    assert_size_stride(add_312, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_57, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_172, (768, ), (1, ))
    assert_size_stride(add_318, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_58, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_175, (768, ), (1, ))
    assert_size_stride(add_323, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_59, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_178, (768, ), (1, ))
    assert_size_stride(add_329, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_60, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_181, (768, ), (1, ))
    assert_size_stride(add_334, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_61, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_184, (768, ), (1, ))
    assert_size_stride(add_340, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_62, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_187, (768, ), (1, ))
    assert_size_stride(add_345, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_63, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_190, (768, ), (1, ))
    assert_size_stride(add_351, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(convolution_64, (8, 768, 32, 32), (786432, 1, 24576, 768))
    assert_size_stride(squeeze_193, (768, ), (1, ))
    assert_size_stride(clone, (8, 768), (768, 1))
    assert_size_stride(permute_1, (1000, 768), (768, 1))
    assert_size_stride(unsqueeze_262, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_274, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_286, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_298, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_310, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_322, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_334, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_346, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_358, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_370, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_382, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_394, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_406, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_418, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_430, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_442, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_454, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_466, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_478, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_490, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_502, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_514, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_526, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_538, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_550, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_562, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_574, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_586, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_598, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_610, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_622, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_634, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_646, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_658, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_670, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_682, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_694, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_706, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_718, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_730, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_742, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_754, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_766, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_778, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_790, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_802, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_814, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_826, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_838, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_850, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_862, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_874, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_886, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_898, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_910, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_922, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_934, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_946, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_958, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_970, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_982, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_994, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1006, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1018, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(unsqueeze_1030, (1, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(tangents_1, (8, 1000), (1000, 1))
    buf0 = empty((8, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 768), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 8), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1, 1000), device='cpu', dtype=torch.float32)
    buf3 = empty((768, ), device='cpu', dtype=torch.float32)
    buf4 = empty((768, ), device='cpu', dtype=torch.float32)
    buf5 = empty((768, ), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_relu_sum_threshold_backward_0(c_void_p(tangents_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(unsqueeze_262.data_ptr()), c_void_p(squeeze_193.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()))
    del buf0
    del convolution_64
    del primals_259
    del squeeze_193
    del tangents_1
    del unsqueeze_262
    # Source Nodes: [l__mod___blocks_31_2], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, add_351, primals_257, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_351
    del primals_257
    buf8 = buf7[0]
    buf9 = buf7[1]
    buf10 = buf7[2]
    del buf7
    buf11 = buf4; del buf4  # reuse
    buf12 = empty((768, ), device='cpu', dtype=torch.float32)
    buf13 = empty((768, ), device='cpu', dtype=torch.float32)
    buf14 = buf6; del buf6  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_1(c_void_p(buf8.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(unsqueeze_274.data_ptr()), c_void_p(squeeze_190.data_ptr()), c_void_p(primals_255.data_ptr()), c_void_p(buf11.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(buf14.data_ptr()))
    del convolution_63
    del primals_255
    del squeeze_190
    del unsqueeze_274
    # Source Nodes: [getattr_getattr_l__mod___blocks___31_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf15 = aten.convolution_backward(buf14, add_345, primals_253, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_345
    del buf14
    del primals_253
    buf16 = buf15[0]
    buf17 = buf15[1]
    buf18 = buf15[2]
    del buf15
    buf19 = buf12; del buf12  # reuse
    buf20 = empty((768, ), device='cpu', dtype=torch.float32)
    buf21 = empty((768, ), device='cpu', dtype=torch.float32)
    buf22 = buf16; del buf16  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_2(c_void_p(buf22.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(unsqueeze_286.data_ptr()), c_void_p(squeeze_187.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(buf20.data_ptr()), c_void_p(buf21.data_ptr()))
    del buf8
    del convolution_62
    del primals_251
    del squeeze_187
    del unsqueeze_286
    # Source Nodes: [l__mod___blocks_30_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf23 = aten.convolution_backward(buf22, add_340, primals_249, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_340
    del primals_249
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = buf20; del buf20  # reuse
    buf28 = empty((768, ), device='cpu', dtype=torch.float32)
    buf29 = empty((768, ), device='cpu', dtype=torch.float32)
    buf30 = buf22; del buf22  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_3(c_void_p(buf24.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(unsqueeze_298.data_ptr()), c_void_p(squeeze_184.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(buf27.data_ptr()), c_void_p(buf28.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()))
    del convolution_61
    del primals_247
    del squeeze_184
    del unsqueeze_298
    # Source Nodes: [getattr_getattr_l__mod___blocks___30_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf31 = aten.convolution_backward(buf30, add_334, primals_245, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_334
    del buf30
    del primals_245
    buf32 = buf31[0]
    buf33 = buf31[1]
    buf34 = buf31[2]
    del buf31
    buf35 = buf28; del buf28  # reuse
    buf36 = empty((768, ), device='cpu', dtype=torch.float32)
    buf37 = empty((768, ), device='cpu', dtype=torch.float32)
    buf38 = buf24; del buf24  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_4(c_void_p(buf38.data_ptr()), c_void_p(buf32.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(unsqueeze_310.data_ptr()), c_void_p(squeeze_181.data_ptr()), c_void_p(primals_243.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()))
    del buf32
    del convolution_60
    del primals_243
    del squeeze_181
    del unsqueeze_310
    # Source Nodes: [l__mod___blocks_29_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf39 = aten.convolution_backward(buf38, add_329, primals_241, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_329
    del primals_241
    buf40 = buf39[0]
    buf41 = buf39[1]
    buf42 = buf39[2]
    del buf39
    buf43 = buf36; del buf36  # reuse
    buf44 = empty((768, ), device='cpu', dtype=torch.float32)
    buf45 = empty((768, ), device='cpu', dtype=torch.float32)
    buf46 = buf38; del buf38  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_5(c_void_p(buf40.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(unsqueeze_322.data_ptr()), c_void_p(squeeze_178.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf43.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf45.data_ptr()), c_void_p(buf46.data_ptr()))
    del convolution_59
    del primals_239
    del squeeze_178
    del unsqueeze_322
    # Source Nodes: [getattr_getattr_l__mod___blocks___29_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf47 = aten.convolution_backward(buf46, add_323, primals_237, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_323
    del buf46
    del primals_237
    buf48 = buf47[0]
    buf49 = buf47[1]
    buf50 = buf47[2]
    del buf47
    buf51 = buf44; del buf44  # reuse
    buf52 = empty((768, ), device='cpu', dtype=torch.float32)
    buf53 = empty((768, ), device='cpu', dtype=torch.float32)
    buf54 = buf40; del buf40  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_6(c_void_p(buf54.data_ptr()), c_void_p(buf48.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(unsqueeze_334.data_ptr()), c_void_p(squeeze_175.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(buf52.data_ptr()), c_void_p(buf53.data_ptr()))
    del buf48
    del convolution_58
    del primals_235
    del squeeze_175
    del unsqueeze_334
    # Source Nodes: [l__mod___blocks_28_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf55 = aten.convolution_backward(buf54, add_318, primals_233, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_318
    del primals_233
    buf56 = buf55[0]
    buf57 = buf55[1]
    buf58 = buf55[2]
    del buf55
    buf59 = buf52; del buf52  # reuse
    buf60 = empty((768, ), device='cpu', dtype=torch.float32)
    buf61 = empty((768, ), device='cpu', dtype=torch.float32)
    buf62 = buf54; del buf54  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_7(c_void_p(buf56.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(unsqueeze_346.data_ptr()), c_void_p(squeeze_172.data_ptr()), c_void_p(primals_231.data_ptr()), c_void_p(buf59.data_ptr()), c_void_p(buf60.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()))
    del convolution_57
    del primals_231
    del squeeze_172
    del unsqueeze_346
    # Source Nodes: [getattr_getattr_l__mod___blocks___28_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf63 = aten.convolution_backward(buf62, add_312, primals_229, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_312
    del buf62
    del primals_229
    buf64 = buf63[0]
    buf65 = buf63[1]
    buf66 = buf63[2]
    del buf63
    buf67 = buf60; del buf60  # reuse
    buf68 = empty((768, ), device='cpu', dtype=torch.float32)
    buf69 = empty((768, ), device='cpu', dtype=torch.float32)
    buf70 = buf56; del buf56  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_8(c_void_p(buf70.data_ptr()), c_void_p(buf64.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(unsqueeze_358.data_ptr()), c_void_p(squeeze_169.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(buf67.data_ptr()), c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()))
    del buf64
    del convolution_56
    del primals_227
    del squeeze_169
    del unsqueeze_358
    # Source Nodes: [l__mod___blocks_27_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf71 = aten.convolution_backward(buf70, add_307, primals_225, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_307
    del primals_225
    buf72 = buf71[0]
    buf73 = buf71[1]
    buf74 = buf71[2]
    del buf71
    buf75 = buf68; del buf68  # reuse
    buf76 = empty((768, ), device='cpu', dtype=torch.float32)
    buf77 = empty((768, ), device='cpu', dtype=torch.float32)
    buf78 = buf70; del buf70  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_9(c_void_p(buf72.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(unsqueeze_370.data_ptr()), c_void_p(squeeze_166.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(buf75.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(buf77.data_ptr()), c_void_p(buf78.data_ptr()))
    del convolution_55
    del primals_223
    del squeeze_166
    del unsqueeze_370
    # Source Nodes: [getattr_getattr_l__mod___blocks___27_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf79 = aten.convolution_backward(buf78, add_301, primals_221, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_301
    del buf78
    del primals_221
    buf80 = buf79[0]
    buf81 = buf79[1]
    buf82 = buf79[2]
    del buf79
    buf83 = buf76; del buf76  # reuse
    buf84 = empty((768, ), device='cpu', dtype=torch.float32)
    buf85 = empty((768, ), device='cpu', dtype=torch.float32)
    buf86 = buf72; del buf72  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_10(c_void_p(buf86.data_ptr()), c_void_p(buf80.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(unsqueeze_382.data_ptr()), c_void_p(squeeze_163.data_ptr()), c_void_p(primals_219.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(buf84.data_ptr()), c_void_p(buf85.data_ptr()))
    del buf80
    del convolution_54
    del primals_219
    del squeeze_163
    del unsqueeze_382
    # Source Nodes: [l__mod___blocks_26_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf87 = aten.convolution_backward(buf86, add_296, primals_217, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_296
    del primals_217
    buf88 = buf87[0]
    buf89 = buf87[1]
    buf90 = buf87[2]
    del buf87
    buf91 = buf84; del buf84  # reuse
    buf92 = empty((768, ), device='cpu', dtype=torch.float32)
    buf93 = empty((768, ), device='cpu', dtype=torch.float32)
    buf94 = buf86; del buf86  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_11(c_void_p(buf88.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(unsqueeze_394.data_ptr()), c_void_p(squeeze_160.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf91.data_ptr()), c_void_p(buf92.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(buf94.data_ptr()))
    del convolution_53
    del primals_215
    del squeeze_160
    del unsqueeze_394
    # Source Nodes: [getattr_getattr_l__mod___blocks___26_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf95 = aten.convolution_backward(buf94, add_290, primals_213, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_290
    del buf94
    del primals_213
    buf96 = buf95[0]
    buf97 = buf95[1]
    buf98 = buf95[2]
    del buf95
    buf99 = buf92; del buf92  # reuse
    buf100 = empty((768, ), device='cpu', dtype=torch.float32)
    buf101 = empty((768, ), device='cpu', dtype=torch.float32)
    buf102 = buf88; del buf88  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_12(c_void_p(buf102.data_ptr()), c_void_p(buf96.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(unsqueeze_406.data_ptr()), c_void_p(squeeze_157.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(buf100.data_ptr()), c_void_p(buf101.data_ptr()))
    del buf96
    del convolution_52
    del primals_211
    del squeeze_157
    del unsqueeze_406
    # Source Nodes: [l__mod___blocks_25_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf103 = aten.convolution_backward(buf102, add_285, primals_209, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_285
    del primals_209
    buf104 = buf103[0]
    buf105 = buf103[1]
    buf106 = buf103[2]
    del buf103
    buf107 = buf100; del buf100  # reuse
    buf108 = empty((768, ), device='cpu', dtype=torch.float32)
    buf109 = empty((768, ), device='cpu', dtype=torch.float32)
    buf110 = buf102; del buf102  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_13(c_void_p(buf104.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(unsqueeze_418.data_ptr()), c_void_p(squeeze_154.data_ptr()), c_void_p(primals_207.data_ptr()), c_void_p(buf107.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(buf109.data_ptr()), c_void_p(buf110.data_ptr()))
    del convolution_51
    del primals_207
    del squeeze_154
    del unsqueeze_418
    # Source Nodes: [getattr_getattr_l__mod___blocks___25_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf111 = aten.convolution_backward(buf110, add_279, primals_205, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_279
    del buf110
    del primals_205
    buf112 = buf111[0]
    buf113 = buf111[1]
    buf114 = buf111[2]
    del buf111
    buf115 = buf108; del buf108  # reuse
    buf116 = empty((768, ), device='cpu', dtype=torch.float32)
    buf117 = empty((768, ), device='cpu', dtype=torch.float32)
    buf118 = buf104; del buf104  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_14(c_void_p(buf118.data_ptr()), c_void_p(buf112.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(unsqueeze_430.data_ptr()), c_void_p(squeeze_151.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(buf116.data_ptr()), c_void_p(buf117.data_ptr()))
    del buf112
    del convolution_50
    del primals_203
    del squeeze_151
    del unsqueeze_430
    # Source Nodes: [l__mod___blocks_24_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf119 = aten.convolution_backward(buf118, add_274, primals_201, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_274
    del primals_201
    buf120 = buf119[0]
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf123 = buf116; del buf116  # reuse
    buf124 = empty((768, ), device='cpu', dtype=torch.float32)
    buf125 = empty((768, ), device='cpu', dtype=torch.float32)
    buf126 = buf118; del buf118  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_15(c_void_p(buf120.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(unsqueeze_442.data_ptr()), c_void_p(squeeze_148.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf123.data_ptr()), c_void_p(buf124.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(buf126.data_ptr()))
    del convolution_49
    del primals_199
    del squeeze_148
    del unsqueeze_442
    # Source Nodes: [getattr_getattr_l__mod___blocks___24_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf127 = aten.convolution_backward(buf126, add_268, primals_197, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_268
    del buf126
    del primals_197
    buf128 = buf127[0]
    buf129 = buf127[1]
    buf130 = buf127[2]
    del buf127
    buf131 = buf124; del buf124  # reuse
    buf132 = empty((768, ), device='cpu', dtype=torch.float32)
    buf133 = empty((768, ), device='cpu', dtype=torch.float32)
    buf134 = buf120; del buf120  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_16(c_void_p(buf134.data_ptr()), c_void_p(buf128.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(unsqueeze_454.data_ptr()), c_void_p(squeeze_145.data_ptr()), c_void_p(primals_195.data_ptr()), c_void_p(buf131.data_ptr()), c_void_p(buf132.data_ptr()), c_void_p(buf133.data_ptr()))
    del buf128
    del convolution_48
    del primals_195
    del squeeze_145
    del unsqueeze_454
    # Source Nodes: [l__mod___blocks_23_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf135 = aten.convolution_backward(buf134, add_263, primals_193, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_263
    del primals_193
    buf136 = buf135[0]
    buf137 = buf135[1]
    buf138 = buf135[2]
    del buf135
    buf139 = buf132; del buf132  # reuse
    buf140 = empty((768, ), device='cpu', dtype=torch.float32)
    buf141 = empty((768, ), device='cpu', dtype=torch.float32)
    buf142 = buf134; del buf134  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_17(c_void_p(buf136.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(unsqueeze_466.data_ptr()), c_void_p(squeeze_142.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(buf141.data_ptr()), c_void_p(buf142.data_ptr()))
    del convolution_47
    del primals_191
    del squeeze_142
    del unsqueeze_466
    # Source Nodes: [getattr_getattr_l__mod___blocks___23_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf143 = aten.convolution_backward(buf142, add_257, primals_189, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_257
    del buf142
    del primals_189
    buf144 = buf143[0]
    buf145 = buf143[1]
    buf146 = buf143[2]
    del buf143
    buf147 = buf140; del buf140  # reuse
    buf148 = empty((768, ), device='cpu', dtype=torch.float32)
    buf149 = empty((768, ), device='cpu', dtype=torch.float32)
    buf150 = buf136; del buf136  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_18(c_void_p(buf150.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(unsqueeze_478.data_ptr()), c_void_p(squeeze_139.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(buf148.data_ptr()), c_void_p(buf149.data_ptr()))
    del buf144
    del convolution_46
    del primals_187
    del squeeze_139
    del unsqueeze_478
    # Source Nodes: [l__mod___blocks_22_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf151 = aten.convolution_backward(buf150, add_252, primals_185, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_252
    del primals_185
    buf152 = buf151[0]
    buf153 = buf151[1]
    buf154 = buf151[2]
    del buf151
    buf155 = buf148; del buf148  # reuse
    buf156 = empty((768, ), device='cpu', dtype=torch.float32)
    buf157 = empty((768, ), device='cpu', dtype=torch.float32)
    buf158 = buf150; del buf150  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_19(c_void_p(buf152.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(unsqueeze_490.data_ptr()), c_void_p(squeeze_136.data_ptr()), c_void_p(primals_183.data_ptr()), c_void_p(buf155.data_ptr()), c_void_p(buf156.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(buf158.data_ptr()))
    del convolution_45
    del primals_183
    del squeeze_136
    del unsqueeze_490
    # Source Nodes: [getattr_getattr_l__mod___blocks___22_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf159 = aten.convolution_backward(buf158, add_246, primals_181, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_246
    del buf158
    del primals_181
    buf160 = buf159[0]
    buf161 = buf159[1]
    buf162 = buf159[2]
    del buf159
    buf163 = buf156; del buf156  # reuse
    buf164 = empty((768, ), device='cpu', dtype=torch.float32)
    buf165 = empty((768, ), device='cpu', dtype=torch.float32)
    buf166 = buf152; del buf152  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_20(c_void_p(buf166.data_ptr()), c_void_p(buf160.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(unsqueeze_502.data_ptr()), c_void_p(squeeze_133.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(buf164.data_ptr()), c_void_p(buf165.data_ptr()))
    del buf160
    del convolution_44
    del primals_179
    del squeeze_133
    del unsqueeze_502
    # Source Nodes: [l__mod___blocks_21_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf167 = aten.convolution_backward(buf166, add_241, primals_177, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_241
    del primals_177
    buf168 = buf167[0]
    buf169 = buf167[1]
    buf170 = buf167[2]
    del buf167
    buf171 = buf164; del buf164  # reuse
    buf172 = empty((768, ), device='cpu', dtype=torch.float32)
    buf173 = empty((768, ), device='cpu', dtype=torch.float32)
    buf174 = buf166; del buf166  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_21(c_void_p(buf168.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(unsqueeze_514.data_ptr()), c_void_p(squeeze_130.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(buf173.data_ptr()), c_void_p(buf174.data_ptr()))
    del convolution_43
    del primals_175
    del squeeze_130
    del unsqueeze_514
    # Source Nodes: [getattr_getattr_l__mod___blocks___21_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf175 = aten.convolution_backward(buf174, add_235, primals_173, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_235
    del buf174
    del primals_173
    buf176 = buf175[0]
    buf177 = buf175[1]
    buf178 = buf175[2]
    del buf175
    buf179 = buf172; del buf172  # reuse
    buf180 = empty((768, ), device='cpu', dtype=torch.float32)
    buf181 = empty((768, ), device='cpu', dtype=torch.float32)
    buf182 = buf168; del buf168  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_22(c_void_p(buf182.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(unsqueeze_526.data_ptr()), c_void_p(squeeze_127.data_ptr()), c_void_p(primals_171.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(buf180.data_ptr()), c_void_p(buf181.data_ptr()))
    del buf176
    del convolution_42
    del primals_171
    del squeeze_127
    del unsqueeze_526
    # Source Nodes: [l__mod___blocks_20_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf183 = aten.convolution_backward(buf182, add_230, primals_169, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_230
    del primals_169
    buf184 = buf183[0]
    buf185 = buf183[1]
    buf186 = buf183[2]
    del buf183
    buf187 = buf180; del buf180  # reuse
    buf188 = empty((768, ), device='cpu', dtype=torch.float32)
    buf189 = empty((768, ), device='cpu', dtype=torch.float32)
    buf190 = buf182; del buf182  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_23(c_void_p(buf184.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(unsqueeze_538.data_ptr()), c_void_p(squeeze_124.data_ptr()), c_void_p(primals_167.data_ptr()), c_void_p(buf187.data_ptr()), c_void_p(buf188.data_ptr()), c_void_p(buf189.data_ptr()), c_void_p(buf190.data_ptr()))
    del convolution_41
    del primals_167
    del squeeze_124
    del unsqueeze_538
    # Source Nodes: [getattr_getattr_l__mod___blocks___20_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf191 = aten.convolution_backward(buf190, add_224, primals_165, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_224
    del buf190
    del primals_165
    buf192 = buf191[0]
    buf193 = buf191[1]
    buf194 = buf191[2]
    del buf191
    buf195 = buf188; del buf188  # reuse
    buf196 = empty((768, ), device='cpu', dtype=torch.float32)
    buf197 = empty((768, ), device='cpu', dtype=torch.float32)
    buf198 = buf184; del buf184  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_24(c_void_p(buf198.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(unsqueeze_550.data_ptr()), c_void_p(squeeze_121.data_ptr()), c_void_p(primals_163.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf197.data_ptr()))
    del buf192
    del convolution_40
    del primals_163
    del squeeze_121
    del unsqueeze_550
    # Source Nodes: [l__mod___blocks_19_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf199 = aten.convolution_backward(buf198, add_219, primals_161, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_219
    del primals_161
    buf200 = buf199[0]
    buf201 = buf199[1]
    buf202 = buf199[2]
    del buf199
    buf203 = buf196; del buf196  # reuse
    buf204 = empty((768, ), device='cpu', dtype=torch.float32)
    buf205 = empty((768, ), device='cpu', dtype=torch.float32)
    buf206 = buf198; del buf198  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_25(c_void_p(buf200.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(unsqueeze_562.data_ptr()), c_void_p(squeeze_118.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf205.data_ptr()), c_void_p(buf206.data_ptr()))
    del convolution_39
    del primals_159
    del squeeze_118
    del unsqueeze_562
    # Source Nodes: [getattr_getattr_l__mod___blocks___19_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf207 = aten.convolution_backward(buf206, add_213, primals_157, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_213
    del buf206
    del primals_157
    buf208 = buf207[0]
    buf209 = buf207[1]
    buf210 = buf207[2]
    del buf207
    buf211 = buf204; del buf204  # reuse
    buf212 = empty((768, ), device='cpu', dtype=torch.float32)
    buf213 = empty((768, ), device='cpu', dtype=torch.float32)
    buf214 = buf200; del buf200  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_26(c_void_p(buf214.data_ptr()), c_void_p(buf208.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(unsqueeze_574.data_ptr()), c_void_p(squeeze_115.data_ptr()), c_void_p(primals_155.data_ptr()), c_void_p(buf211.data_ptr()), c_void_p(buf212.data_ptr()), c_void_p(buf213.data_ptr()))
    del buf208
    del convolution_38
    del primals_155
    del squeeze_115
    del unsqueeze_574
    # Source Nodes: [l__mod___blocks_18_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf215 = aten.convolution_backward(buf214, add_208, primals_153, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_208
    del primals_153
    buf216 = buf215[0]
    buf217 = buf215[1]
    buf218 = buf215[2]
    del buf215
    buf219 = buf212; del buf212  # reuse
    buf220 = empty((768, ), device='cpu', dtype=torch.float32)
    buf221 = empty((768, ), device='cpu', dtype=torch.float32)
    buf222 = buf214; del buf214  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_27(c_void_p(buf216.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(unsqueeze_586.data_ptr()), c_void_p(squeeze_112.data_ptr()), c_void_p(primals_151.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(buf220.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(buf222.data_ptr()))
    del convolution_37
    del primals_151
    del squeeze_112
    del unsqueeze_586
    # Source Nodes: [getattr_getattr_l__mod___blocks___18_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf223 = aten.convolution_backward(buf222, add_202, primals_149, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_202
    del buf222
    del primals_149
    buf224 = buf223[0]
    buf225 = buf223[1]
    buf226 = buf223[2]
    del buf223
    buf227 = buf220; del buf220  # reuse
    buf228 = empty((768, ), device='cpu', dtype=torch.float32)
    buf229 = empty((768, ), device='cpu', dtype=torch.float32)
    buf230 = buf216; del buf216  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_28(c_void_p(buf230.data_ptr()), c_void_p(buf224.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(unsqueeze_598.data_ptr()), c_void_p(squeeze_109.data_ptr()), c_void_p(primals_147.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(buf229.data_ptr()))
    del buf224
    del convolution_36
    del primals_147
    del squeeze_109
    del unsqueeze_598
    # Source Nodes: [l__mod___blocks_17_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf231 = aten.convolution_backward(buf230, add_197, primals_145, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_197
    del primals_145
    buf232 = buf231[0]
    buf233 = buf231[1]
    buf234 = buf231[2]
    del buf231
    buf235 = buf228; del buf228  # reuse
    buf236 = empty((768, ), device='cpu', dtype=torch.float32)
    buf237 = empty((768, ), device='cpu', dtype=torch.float32)
    buf238 = buf230; del buf230  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_29(c_void_p(buf232.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(unsqueeze_610.data_ptr()), c_void_p(squeeze_106.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_35
    del primals_143
    del squeeze_106
    del unsqueeze_610
    # Source Nodes: [getattr_getattr_l__mod___blocks___17_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf239 = aten.convolution_backward(buf238, add_191, primals_141, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_191
    del buf238
    del primals_141
    buf240 = buf239[0]
    buf241 = buf239[1]
    buf242 = buf239[2]
    del buf239
    buf243 = buf236; del buf236  # reuse
    buf244 = empty((768, ), device='cpu', dtype=torch.float32)
    buf245 = empty((768, ), device='cpu', dtype=torch.float32)
    buf246 = buf232; del buf232  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_30(c_void_p(buf246.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(unsqueeze_622.data_ptr()), c_void_p(squeeze_103.data_ptr()), c_void_p(primals_139.data_ptr()), c_void_p(buf243.data_ptr()), c_void_p(buf244.data_ptr()), c_void_p(buf245.data_ptr()))
    del buf240
    del convolution_34
    del primals_139
    del squeeze_103
    del unsqueeze_622
    # Source Nodes: [l__mod___blocks_16_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf247 = aten.convolution_backward(buf246, add_186, primals_137, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_186
    del primals_137
    buf248 = buf247[0]
    buf249 = buf247[1]
    buf250 = buf247[2]
    del buf247
    buf251 = buf244; del buf244  # reuse
    buf252 = empty((768, ), device='cpu', dtype=torch.float32)
    buf253 = empty((768, ), device='cpu', dtype=torch.float32)
    buf254 = buf246; del buf246  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_31(c_void_p(buf248.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(unsqueeze_634.data_ptr()), c_void_p(squeeze_100.data_ptr()), c_void_p(primals_135.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(buf252.data_ptr()), c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_33
    del primals_135
    del squeeze_100
    del unsqueeze_634
    # Source Nodes: [getattr_getattr_l__mod___blocks___16_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf255 = aten.convolution_backward(buf254, add_180, primals_133, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_180
    del buf254
    del primals_133
    buf256 = buf255[0]
    buf257 = buf255[1]
    buf258 = buf255[2]
    del buf255
    buf259 = buf252; del buf252  # reuse
    buf260 = empty((768, ), device='cpu', dtype=torch.float32)
    buf261 = empty((768, ), device='cpu', dtype=torch.float32)
    buf262 = buf248; del buf248  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_32(c_void_p(buf262.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(unsqueeze_646.data_ptr()), c_void_p(squeeze_97.data_ptr()), c_void_p(primals_131.data_ptr()), c_void_p(buf259.data_ptr()), c_void_p(buf260.data_ptr()), c_void_p(buf261.data_ptr()))
    del buf256
    del convolution_32
    del primals_131
    del squeeze_97
    del unsqueeze_646
    # Source Nodes: [l__mod___blocks_15_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf263 = aten.convolution_backward(buf262, add_175, primals_129, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_175
    del primals_129
    buf264 = buf263[0]
    buf265 = buf263[1]
    buf266 = buf263[2]
    del buf263
    buf267 = buf260; del buf260  # reuse
    buf268 = empty((768, ), device='cpu', dtype=torch.float32)
    buf269 = empty((768, ), device='cpu', dtype=torch.float32)
    buf270 = buf262; del buf262  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_33(c_void_p(buf264.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(unsqueeze_658.data_ptr()), c_void_p(squeeze_94.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf267.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf270.data_ptr()))
    del convolution_31
    del primals_127
    del squeeze_94
    del unsqueeze_658
    # Source Nodes: [getattr_getattr_l__mod___blocks___15_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf271 = aten.convolution_backward(buf270, add_169, primals_125, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_169
    del buf270
    del primals_125
    buf272 = buf271[0]
    buf273 = buf271[1]
    buf274 = buf271[2]
    del buf271
    buf275 = buf268; del buf268  # reuse
    buf276 = empty((768, ), device='cpu', dtype=torch.float32)
    buf277 = empty((768, ), device='cpu', dtype=torch.float32)
    buf278 = buf264; del buf264  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_34(c_void_p(buf278.data_ptr()), c_void_p(buf272.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(unsqueeze_670.data_ptr()), c_void_p(squeeze_91.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(buf275.data_ptr()), c_void_p(buf276.data_ptr()), c_void_p(buf277.data_ptr()))
    del buf272
    del convolution_30
    del primals_123
    del squeeze_91
    del unsqueeze_670
    # Source Nodes: [l__mod___blocks_14_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf279 = aten.convolution_backward(buf278, add_164, primals_121, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_164
    del primals_121
    buf280 = buf279[0]
    buf281 = buf279[1]
    buf282 = buf279[2]
    del buf279
    buf283 = buf276; del buf276  # reuse
    buf284 = empty((768, ), device='cpu', dtype=torch.float32)
    buf285 = empty((768, ), device='cpu', dtype=torch.float32)
    buf286 = buf278; del buf278  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_35(c_void_p(buf280.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(unsqueeze_682.data_ptr()), c_void_p(squeeze_88.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf283.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()))
    del convolution_29
    del primals_119
    del squeeze_88
    del unsqueeze_682
    # Source Nodes: [getattr_getattr_l__mod___blocks___14_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf287 = aten.convolution_backward(buf286, add_158, primals_117, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_158
    del buf286
    del primals_117
    buf288 = buf287[0]
    buf289 = buf287[1]
    buf290 = buf287[2]
    del buf287
    buf291 = buf284; del buf284  # reuse
    buf292 = empty((768, ), device='cpu', dtype=torch.float32)
    buf293 = empty((768, ), device='cpu', dtype=torch.float32)
    buf294 = buf280; del buf280  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_36(c_void_p(buf294.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(unsqueeze_694.data_ptr()), c_void_p(squeeze_85.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf291.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(buf293.data_ptr()))
    del buf288
    del convolution_28
    del primals_115
    del squeeze_85
    del unsqueeze_694
    # Source Nodes: [l__mod___blocks_13_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf295 = aten.convolution_backward(buf294, add_153, primals_113, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_153
    del primals_113
    buf296 = buf295[0]
    buf297 = buf295[1]
    buf298 = buf295[2]
    del buf295
    buf299 = buf292; del buf292  # reuse
    buf300 = empty((768, ), device='cpu', dtype=torch.float32)
    buf301 = empty((768, ), device='cpu', dtype=torch.float32)
    buf302 = buf294; del buf294  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_37(c_void_p(buf296.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(unsqueeze_706.data_ptr()), c_void_p(squeeze_82.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf300.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del convolution_27
    del primals_111
    del squeeze_82
    del unsqueeze_706
    # Source Nodes: [getattr_getattr_l__mod___blocks___13_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf303 = aten.convolution_backward(buf302, add_147, primals_109, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_147
    del buf302
    del primals_109
    buf304 = buf303[0]
    buf305 = buf303[1]
    buf306 = buf303[2]
    del buf303
    buf307 = buf300; del buf300  # reuse
    buf308 = empty((768, ), device='cpu', dtype=torch.float32)
    buf309 = empty((768, ), device='cpu', dtype=torch.float32)
    buf310 = buf296; del buf296  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_38(c_void_p(buf310.data_ptr()), c_void_p(buf304.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(unsqueeze_718.data_ptr()), c_void_p(squeeze_79.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf307.data_ptr()), c_void_p(buf308.data_ptr()), c_void_p(buf309.data_ptr()))
    del buf304
    del convolution_26
    del primals_107
    del squeeze_79
    del unsqueeze_718
    # Source Nodes: [l__mod___blocks_12_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf311 = aten.convolution_backward(buf310, add_142, primals_105, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_142
    del primals_105
    buf312 = buf311[0]
    buf313 = buf311[1]
    buf314 = buf311[2]
    del buf311
    buf315 = buf308; del buf308  # reuse
    buf316 = empty((768, ), device='cpu', dtype=torch.float32)
    buf317 = empty((768, ), device='cpu', dtype=torch.float32)
    buf318 = buf310; del buf310  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_39(c_void_p(buf312.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(unsqueeze_730.data_ptr()), c_void_p(squeeze_76.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf315.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(buf317.data_ptr()), c_void_p(buf318.data_ptr()))
    del convolution_25
    del primals_103
    del squeeze_76
    del unsqueeze_730
    # Source Nodes: [getattr_getattr_l__mod___blocks___12_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf319 = aten.convolution_backward(buf318, add_136, primals_101, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_136
    del buf318
    del primals_101
    buf320 = buf319[0]
    buf321 = buf319[1]
    buf322 = buf319[2]
    del buf319
    buf323 = buf316; del buf316  # reuse
    buf324 = empty((768, ), device='cpu', dtype=torch.float32)
    buf325 = empty((768, ), device='cpu', dtype=torch.float32)
    buf326 = buf312; del buf312  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_40(c_void_p(buf326.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(unsqueeze_742.data_ptr()), c_void_p(squeeze_73.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf323.data_ptr()), c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()))
    del buf320
    del convolution_24
    del primals_99
    del squeeze_73
    del unsqueeze_742
    # Source Nodes: [l__mod___blocks_11_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf327 = aten.convolution_backward(buf326, add_131, primals_97, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_131
    del primals_97
    buf328 = buf327[0]
    buf329 = buf327[1]
    buf330 = buf327[2]
    del buf327
    buf331 = buf324; del buf324  # reuse
    buf332 = empty((768, ), device='cpu', dtype=torch.float32)
    buf333 = empty((768, ), device='cpu', dtype=torch.float32)
    buf334 = buf326; del buf326  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_41(c_void_p(buf328.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(unsqueeze_754.data_ptr()), c_void_p(squeeze_70.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(buf332.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf334.data_ptr()))
    del convolution_23
    del primals_95
    del squeeze_70
    del unsqueeze_754
    # Source Nodes: [getattr_getattr_l__mod___blocks___11_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf335 = aten.convolution_backward(buf334, add_125, primals_93, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_125
    del buf334
    del primals_93
    buf336 = buf335[0]
    buf337 = buf335[1]
    buf338 = buf335[2]
    del buf335
    buf339 = buf332; del buf332  # reuse
    buf340 = empty((768, ), device='cpu', dtype=torch.float32)
    buf341 = empty((768, ), device='cpu', dtype=torch.float32)
    buf342 = buf328; del buf328  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_42(c_void_p(buf342.data_ptr()), c_void_p(buf336.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(unsqueeze_766.data_ptr()), c_void_p(squeeze_67.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(buf340.data_ptr()), c_void_p(buf341.data_ptr()))
    del buf336
    del convolution_22
    del primals_91
    del squeeze_67
    del unsqueeze_766
    # Source Nodes: [l__mod___blocks_10_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf343 = aten.convolution_backward(buf342, add_120, primals_89, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_120
    del primals_89
    buf344 = buf343[0]
    buf345 = buf343[1]
    buf346 = buf343[2]
    del buf343
    buf347 = buf340; del buf340  # reuse
    buf348 = empty((768, ), device='cpu', dtype=torch.float32)
    buf349 = empty((768, ), device='cpu', dtype=torch.float32)
    buf350 = buf342; del buf342  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_43(c_void_p(buf344.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(unsqueeze_778.data_ptr()), c_void_p(squeeze_64.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf347.data_ptr()), c_void_p(buf348.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()))
    del convolution_21
    del primals_87
    del squeeze_64
    del unsqueeze_778
    # Source Nodes: [getattr_getattr_l__mod___blocks___10_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf351 = aten.convolution_backward(buf350, add_114, primals_85, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_114
    del buf350
    del primals_85
    buf352 = buf351[0]
    buf353 = buf351[1]
    buf354 = buf351[2]
    del buf351
    buf355 = buf348; del buf348  # reuse
    buf356 = empty((768, ), device='cpu', dtype=torch.float32)
    buf357 = empty((768, ), device='cpu', dtype=torch.float32)
    buf358 = buf344; del buf344  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_44(c_void_p(buf358.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(unsqueeze_790.data_ptr()), c_void_p(squeeze_61.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf355.data_ptr()), c_void_p(buf356.data_ptr()), c_void_p(buf357.data_ptr()))
    del buf352
    del convolution_20
    del primals_83
    del squeeze_61
    del unsqueeze_790
    # Source Nodes: [l__mod___blocks_9_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf359 = aten.convolution_backward(buf358, add_109, primals_81, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_109
    del primals_81
    buf360 = buf359[0]
    buf361 = buf359[1]
    buf362 = buf359[2]
    del buf359
    buf363 = buf356; del buf356  # reuse
    buf364 = empty((768, ), device='cpu', dtype=torch.float32)
    buf365 = empty((768, ), device='cpu', dtype=torch.float32)
    buf366 = buf358; del buf358  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_45(c_void_p(buf360.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(unsqueeze_802.data_ptr()), c_void_p(squeeze_58.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf363.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(buf365.data_ptr()), c_void_p(buf366.data_ptr()))
    del convolution_19
    del primals_79
    del squeeze_58
    del unsqueeze_802
    # Source Nodes: [getattr_getattr_l__mod___blocks___9_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf367 = aten.convolution_backward(buf366, add_103, primals_77, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_103
    del buf366
    del primals_77
    buf368 = buf367[0]
    buf369 = buf367[1]
    buf370 = buf367[2]
    del buf367
    buf371 = buf364; del buf364  # reuse
    buf372 = empty((768, ), device='cpu', dtype=torch.float32)
    buf373 = empty((768, ), device='cpu', dtype=torch.float32)
    buf374 = buf360; del buf360  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_46(c_void_p(buf374.data_ptr()), c_void_p(buf368.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(unsqueeze_814.data_ptr()), c_void_p(squeeze_55.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf372.data_ptr()), c_void_p(buf373.data_ptr()))
    del buf368
    del convolution_18
    del primals_75
    del squeeze_55
    del unsqueeze_814
    # Source Nodes: [l__mod___blocks_8_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf375 = aten.convolution_backward(buf374, add_98, primals_73, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_98
    del primals_73
    buf376 = buf375[0]
    buf377 = buf375[1]
    buf378 = buf375[2]
    del buf375
    buf379 = buf372; del buf372  # reuse
    buf380 = empty((768, ), device='cpu', dtype=torch.float32)
    buf381 = empty((768, ), device='cpu', dtype=torch.float32)
    buf382 = buf374; del buf374  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_47(c_void_p(buf376.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(unsqueeze_826.data_ptr()), c_void_p(squeeze_52.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf379.data_ptr()), c_void_p(buf380.data_ptr()), c_void_p(buf381.data_ptr()), c_void_p(buf382.data_ptr()))
    del convolution_17
    del primals_71
    del squeeze_52
    del unsqueeze_826
    # Source Nodes: [getattr_getattr_l__mod___blocks___8_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf383 = aten.convolution_backward(buf382, add_92, primals_69, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_92
    del buf382
    del primals_69
    buf384 = buf383[0]
    buf385 = buf383[1]
    buf386 = buf383[2]
    del buf383
    buf387 = buf380; del buf380  # reuse
    buf388 = empty((768, ), device='cpu', dtype=torch.float32)
    buf389 = empty((768, ), device='cpu', dtype=torch.float32)
    buf390 = buf376; del buf376  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_48(c_void_p(buf390.data_ptr()), c_void_p(buf384.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(unsqueeze_838.data_ptr()), c_void_p(squeeze_49.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(buf388.data_ptr()), c_void_p(buf389.data_ptr()))
    del buf384
    del convolution_16
    del primals_67
    del squeeze_49
    del unsqueeze_838
    # Source Nodes: [l__mod___blocks_7_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf391 = aten.convolution_backward(buf390, add_87, primals_65, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_87
    del primals_65
    buf392 = buf391[0]
    buf393 = buf391[1]
    buf394 = buf391[2]
    del buf391
    buf395 = buf388; del buf388  # reuse
    buf396 = empty((768, ), device='cpu', dtype=torch.float32)
    buf397 = empty((768, ), device='cpu', dtype=torch.float32)
    buf398 = buf390; del buf390  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_49(c_void_p(buf392.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(unsqueeze_850.data_ptr()), c_void_p(squeeze_46.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf395.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf397.data_ptr()), c_void_p(buf398.data_ptr()))
    del convolution_15
    del primals_63
    del squeeze_46
    del unsqueeze_850
    # Source Nodes: [getattr_getattr_l__mod___blocks___7_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf399 = aten.convolution_backward(buf398, add_81, primals_61, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_81
    del buf398
    del primals_61
    buf400 = buf399[0]
    buf401 = buf399[1]
    buf402 = buf399[2]
    del buf399
    buf403 = buf396; del buf396  # reuse
    buf404 = empty((768, ), device='cpu', dtype=torch.float32)
    buf405 = empty((768, ), device='cpu', dtype=torch.float32)
    buf406 = buf392; del buf392  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_50(c_void_p(buf406.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(unsqueeze_862.data_ptr()), c_void_p(squeeze_43.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf403.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf405.data_ptr()))
    del buf400
    del convolution_14
    del primals_59
    del squeeze_43
    del unsqueeze_862
    # Source Nodes: [l__mod___blocks_6_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf407 = aten.convolution_backward(buf406, add_76, primals_57, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_76
    del primals_57
    buf408 = buf407[0]
    buf409 = buf407[1]
    buf410 = buf407[2]
    del buf407
    buf411 = buf404; del buf404  # reuse
    buf412 = empty((768, ), device='cpu', dtype=torch.float32)
    buf413 = empty((768, ), device='cpu', dtype=torch.float32)
    buf414 = buf406; del buf406  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_51(c_void_p(buf408.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(unsqueeze_874.data_ptr()), c_void_p(squeeze_40.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf411.data_ptr()), c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(buf414.data_ptr()))
    del convolution_13
    del primals_55
    del squeeze_40
    del unsqueeze_874
    # Source Nodes: [getattr_getattr_l__mod___blocks___6_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf415 = aten.convolution_backward(buf414, add_70, primals_53, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_70
    del buf414
    del primals_53
    buf416 = buf415[0]
    buf417 = buf415[1]
    buf418 = buf415[2]
    del buf415
    buf419 = buf412; del buf412  # reuse
    buf420 = empty((768, ), device='cpu', dtype=torch.float32)
    buf421 = empty((768, ), device='cpu', dtype=torch.float32)
    buf422 = buf408; del buf408  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_52(c_void_p(buf422.data_ptr()), c_void_p(buf416.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(unsqueeze_886.data_ptr()), c_void_p(squeeze_37.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(buf420.data_ptr()), c_void_p(buf421.data_ptr()))
    del buf416
    del convolution_12
    del primals_51
    del squeeze_37
    del unsqueeze_886
    # Source Nodes: [l__mod___blocks_5_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf423 = aten.convolution_backward(buf422, add_65, primals_49, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_65
    del primals_49
    buf424 = buf423[0]
    buf425 = buf423[1]
    buf426 = buf423[2]
    del buf423
    buf427 = buf420; del buf420  # reuse
    buf428 = empty((768, ), device='cpu', dtype=torch.float32)
    buf429 = empty((768, ), device='cpu', dtype=torch.float32)
    buf430 = buf422; del buf422  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_53(c_void_p(buf424.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(unsqueeze_898.data_ptr()), c_void_p(squeeze_34.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf427.data_ptr()), c_void_p(buf428.data_ptr()), c_void_p(buf429.data_ptr()), c_void_p(buf430.data_ptr()))
    del convolution_11
    del primals_47
    del squeeze_34
    del unsqueeze_898
    # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf431 = aten.convolution_backward(buf430, add_59, primals_45, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_59
    del buf430
    del primals_45
    buf432 = buf431[0]
    buf433 = buf431[1]
    buf434 = buf431[2]
    del buf431
    buf435 = buf428; del buf428  # reuse
    buf436 = empty((768, ), device='cpu', dtype=torch.float32)
    buf437 = empty((768, ), device='cpu', dtype=torch.float32)
    buf438 = buf424; del buf424  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_54(c_void_p(buf438.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(unsqueeze_910.data_ptr()), c_void_p(squeeze_31.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf435.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf437.data_ptr()))
    del buf432
    del convolution_10
    del primals_43
    del squeeze_31
    del unsqueeze_910
    # Source Nodes: [l__mod___blocks_4_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf439 = aten.convolution_backward(buf438, add_54, primals_41, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_54
    del primals_41
    buf440 = buf439[0]
    buf441 = buf439[1]
    buf442 = buf439[2]
    del buf439
    buf443 = buf436; del buf436  # reuse
    buf444 = empty((768, ), device='cpu', dtype=torch.float32)
    buf445 = empty((768, ), device='cpu', dtype=torch.float32)
    buf446 = buf438; del buf438  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_55(c_void_p(buf440.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(unsqueeze_922.data_ptr()), c_void_p(squeeze_28.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf443.data_ptr()), c_void_p(buf444.data_ptr()), c_void_p(buf445.data_ptr()), c_void_p(buf446.data_ptr()))
    del convolution_9
    del primals_39
    del squeeze_28
    del unsqueeze_922
    # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf447 = aten.convolution_backward(buf446, add_48, primals_37, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_48
    del buf446
    del primals_37
    buf448 = buf447[0]
    buf449 = buf447[1]
    buf450 = buf447[2]
    del buf447
    buf451 = buf444; del buf444  # reuse
    buf452 = empty((768, ), device='cpu', dtype=torch.float32)
    buf453 = empty((768, ), device='cpu', dtype=torch.float32)
    buf454 = buf440; del buf440  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_56(c_void_p(buf454.data_ptr()), c_void_p(buf448.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(unsqueeze_934.data_ptr()), c_void_p(squeeze_25.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf451.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(buf453.data_ptr()))
    del buf448
    del convolution_8
    del primals_35
    del squeeze_25
    del unsqueeze_934
    # Source Nodes: [l__mod___blocks_3_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf455 = aten.convolution_backward(buf454, add_43, primals_33, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_43
    del primals_33
    buf456 = buf455[0]
    buf457 = buf455[1]
    buf458 = buf455[2]
    del buf455
    buf459 = buf452; del buf452  # reuse
    buf460 = empty((768, ), device='cpu', dtype=torch.float32)
    buf461 = empty((768, ), device='cpu', dtype=torch.float32)
    buf462 = buf454; del buf454  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_57(c_void_p(buf456.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(unsqueeze_946.data_ptr()), c_void_p(squeeze_22.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf459.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(buf461.data_ptr()), c_void_p(buf462.data_ptr()))
    del convolution_7
    del primals_31
    del squeeze_22
    del unsqueeze_946
    # Source Nodes: [getattr_getattr_l__mod___blocks___3_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf463 = aten.convolution_backward(buf462, add_37, primals_29, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_37
    del buf462
    del primals_29
    buf464 = buf463[0]
    buf465 = buf463[1]
    buf466 = buf463[2]
    del buf463
    buf467 = buf460; del buf460  # reuse
    buf468 = empty((768, ), device='cpu', dtype=torch.float32)
    buf469 = empty((768, ), device='cpu', dtype=torch.float32)
    buf470 = buf456; del buf456  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_58(c_void_p(buf470.data_ptr()), c_void_p(buf464.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(unsqueeze_958.data_ptr()), c_void_p(squeeze_19.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf468.data_ptr()), c_void_p(buf469.data_ptr()))
    del buf464
    del convolution_6
    del primals_27
    del squeeze_19
    del unsqueeze_958
    # Source Nodes: [l__mod___blocks_2_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf471 = aten.convolution_backward(buf470, add_32, primals_25, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_32
    del primals_25
    buf472 = buf471[0]
    buf473 = buf471[1]
    buf474 = buf471[2]
    del buf471
    buf475 = buf468; del buf468  # reuse
    buf476 = empty((768, ), device='cpu', dtype=torch.float32)
    buf477 = empty((768, ), device='cpu', dtype=torch.float32)
    buf478 = buf470; del buf470  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_59(c_void_p(buf472.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(unsqueeze_970.data_ptr()), c_void_p(squeeze_16.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf475.data_ptr()), c_void_p(buf476.data_ptr()), c_void_p(buf477.data_ptr()), c_void_p(buf478.data_ptr()))
    del convolution_5
    del primals_23
    del squeeze_16
    del unsqueeze_970
    # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf479 = aten.convolution_backward(buf478, add_26, primals_21, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_26
    del buf478
    del primals_21
    buf480 = buf479[0]
    buf481 = buf479[1]
    buf482 = buf479[2]
    del buf479
    buf483 = buf476; del buf476  # reuse
    buf484 = empty((768, ), device='cpu', dtype=torch.float32)
    buf485 = empty((768, ), device='cpu', dtype=torch.float32)
    buf486 = buf472; del buf472  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_60(c_void_p(buf486.data_ptr()), c_void_p(buf480.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(unsqueeze_982.data_ptr()), c_void_p(squeeze_13.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf483.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(buf485.data_ptr()))
    del buf480
    del convolution_4
    del primals_19
    del squeeze_13
    del unsqueeze_982
    # Source Nodes: [l__mod___blocks_1_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf487 = aten.convolution_backward(buf486, add_21, primals_17, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_21
    del primals_17
    buf488 = buf487[0]
    buf489 = buf487[1]
    buf490 = buf487[2]
    del buf487
    buf491 = buf484; del buf484  # reuse
    buf492 = empty((768, ), device='cpu', dtype=torch.float32)
    buf493 = empty((768, ), device='cpu', dtype=torch.float32)
    buf494 = buf486; del buf486  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_61(c_void_p(buf488.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(unsqueeze_994.data_ptr()), c_void_p(squeeze_10.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf491.data_ptr()), c_void_p(buf492.data_ptr()), c_void_p(buf493.data_ptr()), c_void_p(buf494.data_ptr()))
    del convolution_3
    del primals_15
    del squeeze_10
    del unsqueeze_994
    # Source Nodes: [getattr_getattr_l__mod___blocks___1_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf495 = aten.convolution_backward(buf494, add_15, primals_13, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_15
    del buf494
    del primals_13
    buf496 = buf495[0]
    buf497 = buf495[1]
    buf498 = buf495[2]
    del buf495
    buf499 = buf492; del buf492  # reuse
    buf500 = empty((768, ), device='cpu', dtype=torch.float32)
    buf501 = empty((768, ), device='cpu', dtype=torch.float32)
    buf502 = buf488; del buf488  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_62(c_void_p(buf502.data_ptr()), c_void_p(buf496.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(unsqueeze_1006.data_ptr()), c_void_p(squeeze_7.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(buf500.data_ptr()), c_void_p(buf501.data_ptr()))
    del buf496
    del convolution_2
    del primals_11
    del squeeze_7
    del unsqueeze_1006
    # Source Nodes: [l__mod___blocks_0_2], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf503 = aten.convolution_backward(buf502, add_10, primals_9, [768], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del add_10
    del primals_9
    buf504 = buf503[0]
    buf505 = buf503[1]
    buf506 = buf503[2]
    del buf503
    buf507 = buf500; del buf500  # reuse
    buf508 = empty((768, ), device='cpu', dtype=torch.float32)
    buf509 = empty((768, ), device='cpu', dtype=torch.float32)
    buf510 = buf502; del buf502  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_relu_threshold_backward_63(c_void_p(buf504.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(unsqueeze_1018.data_ptr()), c_void_p(squeeze_4.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(buf508.data_ptr()), c_void_p(buf509.data_ptr()), c_void_p(buf510.data_ptr()))
    del convolution_1
    del primals_7
    del squeeze_4
    del unsqueeze_1018
    # Source Nodes: [getattr_getattr_l__mod___blocks___0_____0___fn_1], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf511 = aten.convolution_backward(buf510, add_4, primals_5, [768], [1, 1], [3, 3], [1, 1], False, [0, 0], 768, [True, True, True])
    del add_4
    del buf510
    del primals_5
    buf512 = buf511[0]
    buf513 = buf511[1]
    buf514 = buf511[2]
    del buf511
    buf515 = buf508; del buf508  # reuse
    buf516 = empty((768, ), device='cpu', dtype=torch.float32)
    buf517 = empty((768, ), device='cpu', dtype=torch.float32)
    buf518 = buf504; del buf504  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_relu_threshold_backward_64(c_void_p(buf518.data_ptr()), c_void_p(buf512.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(unsqueeze_1030.data_ptr()), c_void_p(squeeze_1.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf515.data_ptr()), c_void_p(buf516.data_ptr()), c_void_p(buf517.data_ptr()))
    del buf512
    del buf516
    del convolution
    del primals_3
    del squeeze_1
    del unsqueeze_1030
    # Source Nodes: [l__mod___stem_1], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.relu, aten.threshold_backward]
    buf519 = aten.convolution_backward(buf518, primals_458, primals_1, [768], [7, 7], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True])
    del buf518
    del primals_1
    del primals_458
    buf520 = buf519[1]
    buf521 = buf519[2]
    return (buf520, buf521, buf517, buf515, buf513, buf514, buf509, buf507, buf505, buf506, buf501, buf499, buf497, buf498, buf493, buf491, buf489, buf490, buf485, buf483, buf481, buf482, buf477, buf475, buf473, buf474, buf469, buf467, buf465, buf466, buf461, buf459, buf457, buf458, buf453, buf451, buf449, buf450, buf445, buf443, buf441, buf442, buf437, buf435, buf433, buf434, buf429, buf427, buf425, buf426, buf421, buf419, buf417, buf418, buf413, buf411, buf409, buf410, buf405, buf403, buf401, buf402, buf397, buf395, buf393, buf394, buf389, buf387, buf385, buf386, buf381, buf379, buf377, buf378, buf373, buf371, buf369, buf370, buf365, buf363, buf361, buf362, buf357, buf355, buf353, buf354, buf349, buf347, buf345, buf346, buf341, buf339, buf337, buf338, buf333, buf331, buf329, buf330, buf325, buf323, buf321, buf322, buf317, buf315, buf313, buf314, buf309, buf307, buf305, buf306, buf301, buf299, buf297, buf298, buf293, buf291, buf289, buf290, buf285, buf283, buf281, buf282, buf277, buf275, buf273, buf274, buf269, buf267, buf265, buf266, buf261, buf259, buf257, buf258, buf253, buf251, buf249, buf250, buf245, buf243, buf241, buf242, buf237, buf235, buf233, buf234, buf229, buf227, buf225, buf226, buf221, buf219, buf217, buf218, buf213, buf211, buf209, buf210, buf205, buf203, buf201, buf202, buf197, buf195, buf193, buf194, buf189, buf187, buf185, buf186, buf181, buf179, buf177, buf178, buf173, buf171, buf169, buf170, buf165, buf163, buf161, buf162, buf157, buf155, buf153, buf154, buf149, buf147, buf145, buf146, buf141, buf139, buf137, buf138, buf133, buf131, buf129, buf130, buf125, buf123, buf121, buf122, buf117, buf115, buf113, buf114, buf109, buf107, buf105, buf106, buf101, buf99, buf97, buf98, buf93, buf91, buf89, buf90, buf85, buf83, buf81, buf82, buf77, buf75, buf73, buf74, buf69, buf67, buf65, buf66, buf61, buf59, buf57, buf58, buf53, buf51, buf49, buf50, buf45, buf43, buf41, buf42, buf37, buf35, buf33, buf34, buf29, buf27, buf25, buf26, buf21, buf19, buf17, buf18, buf13, buf11, buf9, buf10, buf5, buf3, reinterpret_tensor(buf1, (1000, 768), (768, 1), 0), reinterpret_tensor(buf2, (1000, ), (1, ), 0), None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((768, 3, 7, 7), (147, 1, 21, 3), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_173 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_195 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_225 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_237 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((768, 1, 7, 7), (49, 49, 7, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((768, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    primals_458 = rand_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_1 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_4 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_4 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_10 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_7 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_15 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_10 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_21 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_13 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_26 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_16 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_32 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_19 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_37 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_22 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_43 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_25 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_48 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_28 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_54 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_31 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_59 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_34 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_65 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_37 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_70 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_40 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_76 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_43 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_81 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_46 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_87 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_49 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_92 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_52 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_98 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_55 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_103 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_58 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_109 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_61 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_114 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_64 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_120 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_67 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_125 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_70 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_131 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_73 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_136 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_76 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_142 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_79 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_147 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_82 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_153 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_85 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_158 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_88 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_164 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_91 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_169 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_94 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_175 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_97 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_180 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_100 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_186 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_103 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_191 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_106 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_197 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_109 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_202 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_112 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_208 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_115 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_213 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_118 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_219 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_121 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_224 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_124 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_230 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_127 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_235 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_130 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_241 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_133 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_246 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_136 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_252 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_139 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_257 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_142 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_263 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_145 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_268 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_148 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_274 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_151 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_279 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_154 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_285 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_157 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_290 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_160 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_296 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_163 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_301 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_166 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_307 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_169 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_312 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_172 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_318 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_175 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_323 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_178 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_329 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_181 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_334 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_184 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_340 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_187 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_345 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_190 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    add_351 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((8, 768, 32, 32), (786432, 1, 24576, 768), device='cpu', dtype=torch.float32)
    squeeze_193 = rand_strided((768, ), (1, ), device='cpu', dtype=torch.float32)
    clone = rand_strided((8, 768), (768, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 768), (768, 1), device='cpu', dtype=torch.float32)
    unsqueeze_262 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_274 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_286 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_298 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_310 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_322 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_334 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_346 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_358 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_370 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_382 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_394 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_406 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_418 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_430 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_442 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_454 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_466 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_478 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_490 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_502 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_514 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_526 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_538 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_550 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_562 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_574 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_586 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_598 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_610 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_622 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_634 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_646 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_658 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_670 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_682 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_694 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_706 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_718 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_730 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_742 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_754 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_766 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_778 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_790 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_802 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_814 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_826 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_838 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_850 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_862 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_874 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_886 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_898 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_910 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_922 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_934 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_946 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_958 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_970 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_982 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_994 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1006 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1018 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    unsqueeze_1030 = rand_strided((1, 768, 1, 1), (768, 1, 1, 1), device='cpu', dtype=torch.float32)
    tangents_1 = rand_strided((8, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_225, primals_227, primals_229, primals_231, primals_233, primals_235, primals_237, primals_239, primals_241, primals_243, primals_245, primals_247, primals_249, primals_251, primals_253, primals_255, primals_257, primals_259, primals_458, convolution, squeeze_1, add_4, convolution_1, squeeze_4, add_10, convolution_2, squeeze_7, add_15, convolution_3, squeeze_10, add_21, convolution_4, squeeze_13, add_26, convolution_5, squeeze_16, add_32, convolution_6, squeeze_19, add_37, convolution_7, squeeze_22, add_43, convolution_8, squeeze_25, add_48, convolution_9, squeeze_28, add_54, convolution_10, squeeze_31, add_59, convolution_11, squeeze_34, add_65, convolution_12, squeeze_37, add_70, convolution_13, squeeze_40, add_76, convolution_14, squeeze_43, add_81, convolution_15, squeeze_46, add_87, convolution_16, squeeze_49, add_92, convolution_17, squeeze_52, add_98, convolution_18, squeeze_55, add_103, convolution_19, squeeze_58, add_109, convolution_20, squeeze_61, add_114, convolution_21, squeeze_64, add_120, convolution_22, squeeze_67, add_125, convolution_23, squeeze_70, add_131, convolution_24, squeeze_73, add_136, convolution_25, squeeze_76, add_142, convolution_26, squeeze_79, add_147, convolution_27, squeeze_82, add_153, convolution_28, squeeze_85, add_158, convolution_29, squeeze_88, add_164, convolution_30, squeeze_91, add_169, convolution_31, squeeze_94, add_175, convolution_32, squeeze_97, add_180, convolution_33, squeeze_100, add_186, convolution_34, squeeze_103, add_191, convolution_35, squeeze_106, add_197, convolution_36, squeeze_109, add_202, convolution_37, squeeze_112, add_208, convolution_38, squeeze_115, add_213, convolution_39, squeeze_118, add_219, convolution_40, squeeze_121, add_224, convolution_41, squeeze_124, add_230, convolution_42, squeeze_127, add_235, convolution_43, squeeze_130, add_241, convolution_44, squeeze_133, add_246, convolution_45, squeeze_136, add_252, convolution_46, squeeze_139, add_257, convolution_47, squeeze_142, add_263, convolution_48, squeeze_145, add_268, convolution_49, squeeze_148, add_274, convolution_50, squeeze_151, add_279, convolution_51, squeeze_154, add_285, convolution_52, squeeze_157, add_290, convolution_53, squeeze_160, add_296, convolution_54, squeeze_163, add_301, convolution_55, squeeze_166, add_307, convolution_56, squeeze_169, add_312, convolution_57, squeeze_172, add_318, convolution_58, squeeze_175, add_323, convolution_59, squeeze_178, add_329, convolution_60, squeeze_181, add_334, convolution_61, squeeze_184, add_340, convolution_62, squeeze_187, add_345, convolution_63, squeeze_190, add_351, convolution_64, squeeze_193, clone, permute_1, unsqueeze_262, unsqueeze_274, unsqueeze_286, unsqueeze_298, unsqueeze_310, unsqueeze_322, unsqueeze_334, unsqueeze_346, unsqueeze_358, unsqueeze_370, unsqueeze_382, unsqueeze_394, unsqueeze_406, unsqueeze_418, unsqueeze_430, unsqueeze_442, unsqueeze_454, unsqueeze_466, unsqueeze_478, unsqueeze_490, unsqueeze_502, unsqueeze_514, unsqueeze_526, unsqueeze_538, unsqueeze_550, unsqueeze_562, unsqueeze_574, unsqueeze_586, unsqueeze_598, unsqueeze_610, unsqueeze_622, unsqueeze_634, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_682, unsqueeze_694, unsqueeze_706, unsqueeze_718, unsqueeze_730, unsqueeze_742, unsqueeze_754, unsqueeze_766, unsqueeze_778, unsqueeze_790, unsqueeze_802, unsqueeze_814, unsqueeze_826, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_874, unsqueeze_886, unsqueeze_898, unsqueeze_910, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_958, unsqueeze_970, unsqueeze_982, unsqueeze_994, unsqueeze_1006, unsqueeze_1018, unsqueeze_1030, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('convmixer_768_32', benchmark_compiled_module)
