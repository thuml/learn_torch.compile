
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


cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const bool* in_ptr1,
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
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5)
{
    auto out_ptr2 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1000L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1000L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2000L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3000L + x0));
            auto tmp2 = tmp0 + tmp1;
            auto tmp4 = tmp2 + tmp3;
            auto tmp6 = tmp4 + tmp5;
            tmp6.store(out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2240L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                            auto tmp2 = static_cast<float>(49.0);
                            auto tmp3 = at::vec::Vectorized<float>(tmp2);
                            auto tmp4 = tmp1 / tmp3;
                            auto tmp5 = static_cast<float>(0.0);
                            auto tmp6 = at::vec::Vectorized<float>(tmp5);
                            auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                            auto tmp10 = tmp8 - tmp9;
                            auto tmp11 = tmp7 * tmp10;
                            auto tmp14 = tmp12 - tmp13;
                            auto tmp15 = tmp7 * tmp14;
                            tmp_acc0_vec = tmp_acc0_vec + tmp7;
                            tmp_acc1_vec = tmp_acc1_vec + tmp11;
                            tmp_acc2_vec = tmp_acc2_vec + tmp15;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = flag_to_float_vec(in_ptr1 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2240L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x2));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x2));
                        auto tmp2 = static_cast<float>(49.0);
                        auto tmp3 = at::vec::Vectorized<float>(tmp2);
                        auto tmp4 = tmp1 / tmp3;
                        auto tmp5 = static_cast<float>(0.0);
                        auto tmp6 = at::vec::Vectorized<float>(tmp5);
                        auto tmp7 = decltype(tmp6)::blendv(tmp4, tmp6, tmp0);
                        auto tmp9 = static_cast<float>(1e-05);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp11.rsqrt();
                        auto tmp14 = tmp12 * tmp13;
                        auto tmp15 = tmp7 * tmp14;
                        auto tmp17 = tmp16 + tmp10;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp7 * tmp20;
                        tmp15.store(out_ptr4 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                        tmp21.store(out_ptr5 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_1 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2240L*x2) + (109760L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (2240L*x2) + (109760L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(8960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2240L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (2240L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (2240L*x2) + (109760L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(49.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(2240L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (2240L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (2240L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(49.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (2240L*x1) + (109760L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (2240L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (2240L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (2240L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(2240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (2240L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (2240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_12 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_13 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_14 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_17 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_25 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_26 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_27 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
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
                       const float* in_ptr7,
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
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
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_34 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_38 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_39 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_40 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_42 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_46 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_47 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
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
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_52 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_55 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_56 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_59 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(702464L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_64 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_65 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_66 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_68 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp18 = tmp16 - tmp17;
                        auto tmp19 = tmp11 * tmp18;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                        tmp_acc4_vec = tmp_acc4_vec + tmp19;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_69 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp16 + tmp10;
                    auto tmp18 = tmp17.rsqrt();
                    auto tmp20 = tmp18 * tmp19;
                    auto tmp21 = tmp7 * tmp20;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    tmp21.store(out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_73 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (896L*x2) + (175616L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3584L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_74 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_75 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (896L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (896L*x2) + (175616L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(196.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(896L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (896L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(196.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (896L*x1) + (175616L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (896L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (896L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(896L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (896L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_79 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_80 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(784.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(784.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1404928L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_83 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_84 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_85 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(784.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_87 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_88 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(784.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_89 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_90 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_91 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_92 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_93 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(784.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(784.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_95 = async_compile.cpp('''
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
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(1404928L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_96 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_97 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_98 = async_compile.cpp('''
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
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(784.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(out_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_threshold_backward_100 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc3 = 0;
                    at::vec::Vectorized<float> tmp_acc3_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc4 = 0;
                    at::vec::Vectorized<float> tmp_acc4_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(0.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 <= tmp7);
                        auto tmp10 = tmp0 + tmp9;
                        auto tmp11 = decltype(tmp7)::blendv(tmp10, tmp7, tmp8);
                        auto tmp14 = tmp12 - tmp13;
                        auto tmp15 = tmp11 * tmp14;
                        auto tmp18 = tmp16 - tmp17;
                        auto tmp19 = tmp11 * tmp18;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp11;
                        tmp_acc3_vec = tmp_acc3_vec + tmp15;
                        tmp_acc4_vec = tmp_acc4_vec + tmp19;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                    tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                    tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                    tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_101 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(784.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_threshold_backward_102 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_103 = async_compile.cpp('''
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
                       float* out_ptr1)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(3136L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp16 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    auto tmp17 = tmp16 + tmp10;
                    auto tmp18 = tmp17.rsqrt();
                    auto tmp20 = tmp18 * tmp19;
                    auto tmp21 = tmp7 * tmp20;
                    tmp15.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    tmp21.store(out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_104 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_105 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (448L*x2) + (351232L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1792L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_106 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_107 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (448L*x2) + (351232L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(784.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(448L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (448L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(784.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (448L*x1) + (351232L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_108 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (448L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (448L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(448L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(448L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (448L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_109 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (224L*x1)));
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(out_ptr2 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_110 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_111 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_112 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (224L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(3136.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(3136.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_113 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_114 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4)
{
    auto out_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2809856L); x0+=static_cast<long>(8L))
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc2 = 0;
                    at::vec::Vectorized<float> tmp_acc2_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (224L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x1));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x1));
                    auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp9 + tmp3;
                    auto tmp11 = tmp10.rsqrt();
                    auto tmp13 = tmp11 * tmp12;
                    auto tmp14 = tmp0 * tmp13;
                    tmp8.store(out_ptr3 + static_cast<long>(x1 + (224L*x0)));
                    tmp14.store(out_ptr4 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_native_batch_norm_backward_115 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(1e-05);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_116 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (224L*x2) + (702464L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(896L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = decltype(tmp1)(1)/(decltype(tmp1)(1) + tmp1.neg().exp());
                    auto tmp3 = static_cast<float>(1.0);
                    auto tmp4 = at::vec::Vectorized<float>(tmp3);
                    auto tmp5 = tmp4 - tmp2;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp7 = tmp0 * tmp6;
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_threshold_backward_117 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(0.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 <= tmp2);
            auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
            tmp5.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_118 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       const float* in_ptr6,
                       const float* in_ptr7,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(4L); x1+=static_cast<long>(1L))
                    {
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(3136L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (224L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (224L*x2) + (702464L*x1)));
                            auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                            auto tmp7 = tmp4 * tmp6;
                            auto tmp9 = static_cast<float>(3136.0);
                            auto tmp10 = at::vec::Vectorized<float>(tmp9);
                            auto tmp11 = tmp8 / tmp10;
                            auto tmp12 = tmp7 + tmp11;
                            auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                            auto tmp16 = tmp14 - tmp15;
                            auto tmp17 = tmp13 * tmp16;
                            tmp_acc0_vec = tmp_acc0_vec + tmp13;
                            tmp_acc1_vec = tmp_acc1_vec + tmp17;
                        }
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for  collapse(2)
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(224L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (224L*x0)));
                        auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = decltype(tmp5)(1)/(decltype(tmp5)(1) + tmp5.neg().exp());
                        auto tmp7 = tmp4 * tmp6;
                        auto tmp9 = static_cast<float>(3136.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 / tmp10;
                        auto tmp12 = tmp7 + tmp11;
                        auto tmp13 = decltype(tmp2)::blendv(tmp12, tmp2, tmp3);
                        auto tmp15 = static_cast<float>(1e-05);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp17.rsqrt();
                        auto tmp20 = tmp18 * tmp19;
                        auto tmp21 = tmp13 * tmp20;
                        tmp21.store(in_out_ptr1 + static_cast<long>(x2 + (224L*x1) + (702464L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_119 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       const float* in_ptr5,
                       float* out_ptr0)
{
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (224L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (224L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(224L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(224L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(1e-05);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (224L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_120 = async_compile.cpp('''
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
    auto out_ptr1 = in_out_ptr0;
    auto in_ptr1 = in_out_ptr1;
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
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
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(32L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(1e-05);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp6 = tmp0 * tmp5;
                    tmp6.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(32L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (32L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp7 = decltype(tmp2)::blendv(tmp6, tmp2, tmp3);
                    auto tmp9 = static_cast<float>(1e-05);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp11.rsqrt();
                    auto tmp14 = tmp12 * tmp13;
                    auto tmp15 = tmp7 * tmp14;
                    tmp15.store(in_out_ptr1 + static_cast<long>(x1 + (32L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, mean, relu_3, convolution_4, mul_9, convolution_5, convolution_6, relu_4, convolution_7, relu_5, convolution_8, relu_6, mean_1, relu_7, convolution_10, mul_22, convolution_11, relu_8, convolution_12, relu_9, convolution_13, relu_10, mean_2, relu_11, convolution_15, mul_32, convolution_16, convolution_17, relu_12, convolution_18, relu_13, convolution_19, relu_14, mean_3, relu_15, convolution_21, mul_45, convolution_22, relu_16, convolution_23, relu_17, convolution_24, relu_18, mean_4, relu_19, convolution_26, mul_55, convolution_27, relu_20, convolution_28, relu_21, convolution_29, relu_22, mean_5, relu_23, convolution_31, mul_65, convolution_32, relu_24, convolution_33, relu_25, convolution_34, relu_26, mean_6, relu_27, convolution_36, mul_75, convolution_37, relu_28, convolution_38, relu_29, convolution_39, relu_30, mean_7, relu_31, convolution_41, mul_85, convolution_42, convolution_43, relu_32, convolution_44, relu_33, convolution_45, relu_34, mean_8, relu_35, convolution_47, mul_98, convolution_48, relu_36, convolution_49, relu_37, convolution_50, relu_38, mean_9, relu_39, convolution_52, mul_108, convolution_53, relu_40, convolution_54, relu_41, convolution_55, relu_42, mean_10, relu_43, convolution_57, mul_118, convolution_58, relu_44, convolution_59, relu_45, convolution_60, relu_46, mean_11, relu_47, convolution_62, mul_128, convolution_63, relu_48, convolution_64, relu_49, convolution_65, relu_50, mean_12, relu_51, convolution_67, mul_138, convolution_68, relu_52, convolution_69, relu_53, convolution_70, relu_54, mean_13, relu_55, convolution_72, mul_148, convolution_73, relu_56, convolution_74, relu_57, convolution_75, relu_58, mean_14, relu_59, convolution_77, mul_158, convolution_78, relu_60, convolution_79, relu_61, convolution_80, relu_62, mean_15, relu_63, convolution_82, mul_168, convolution_83, relu_64, convolution_84, relu_65, convolution_85, relu_66, mean_16, relu_67, convolution_87, mul_178, convolution_88, relu_68, convolution_89, relu_69, convolution_90, relu_70, mean_17, relu_71, convolution_92, mul_188, convolution_93, relu_72, convolution_94, relu_73, convolution_95, relu_74, mean_18, relu_75, convolution_97, mul_198, convolution_98, convolution_99, clone, permute_1, le, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (32, ), (1, ))
    assert_size_stride(primals_3, (224, ), (1, ))
    assert_size_stride(primals_5, (224, ), (1, ))
    assert_size_stride(primals_7, (224, ), (1, ))
    assert_size_stride(primals_9, (224, ), (1, ))
    assert_size_stride(primals_11, (224, ), (1, ))
    assert_size_stride(primals_13, (224, ), (1, ))
    assert_size_stride(primals_15, (224, ), (1, ))
    assert_size_stride(primals_17, (448, ), (1, ))
    assert_size_stride(primals_19, (448, ), (1, ))
    assert_size_stride(primals_21, (448, ), (1, ))
    assert_size_stride(primals_23, (448, ), (1, ))
    assert_size_stride(primals_25, (448, ), (1, ))
    assert_size_stride(primals_27, (448, ), (1, ))
    assert_size_stride(primals_29, (448, ), (1, ))
    assert_size_stride(primals_31, (448, ), (1, ))
    assert_size_stride(primals_33, (448, ), (1, ))
    assert_size_stride(primals_35, (448, ), (1, ))
    assert_size_stride(primals_37, (448, ), (1, ))
    assert_size_stride(primals_39, (448, ), (1, ))
    assert_size_stride(primals_41, (448, ), (1, ))
    assert_size_stride(primals_43, (448, ), (1, ))
    assert_size_stride(primals_45, (448, ), (1, ))
    assert_size_stride(primals_47, (448, ), (1, ))
    assert_size_stride(primals_49, (896, ), (1, ))
    assert_size_stride(primals_51, (896, ), (1, ))
    assert_size_stride(primals_53, (896, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_57, (896, ), (1, ))
    assert_size_stride(primals_59, (896, ), (1, ))
    assert_size_stride(primals_61, (896, ), (1, ))
    assert_size_stride(primals_63, (896, ), (1, ))
    assert_size_stride(primals_65, (896, ), (1, ))
    assert_size_stride(primals_67, (896, ), (1, ))
    assert_size_stride(primals_69, (896, ), (1, ))
    assert_size_stride(primals_71, (896, ), (1, ))
    assert_size_stride(primals_73, (896, ), (1, ))
    assert_size_stride(primals_75, (896, ), (1, ))
    assert_size_stride(primals_77, (896, ), (1, ))
    assert_size_stride(primals_79, (896, ), (1, ))
    assert_size_stride(primals_81, (896, ), (1, ))
    assert_size_stride(primals_83, (896, ), (1, ))
    assert_size_stride(primals_85, (896, ), (1, ))
    assert_size_stride(primals_87, (896, ), (1, ))
    assert_size_stride(primals_89, (896, ), (1, ))
    assert_size_stride(primals_91, (896, ), (1, ))
    assert_size_stride(primals_93, (896, ), (1, ))
    assert_size_stride(primals_95, (896, ), (1, ))
    assert_size_stride(primals_97, (896, ), (1, ))
    assert_size_stride(primals_99, (896, ), (1, ))
    assert_size_stride(primals_101, (896, ), (1, ))
    assert_size_stride(primals_103, (896, ), (1, ))
    assert_size_stride(primals_105, (896, ), (1, ))
    assert_size_stride(primals_107, (896, ), (1, ))
    assert_size_stride(primals_109, (896, ), (1, ))
    assert_size_stride(primals_111, (896, ), (1, ))
    assert_size_stride(primals_113, (896, ), (1, ))
    assert_size_stride(primals_115, (896, ), (1, ))
    assert_size_stride(primals_117, (2240, ), (1, ))
    assert_size_stride(primals_119, (2240, ), (1, ))
    assert_size_stride(primals_121, (2240, ), (1, ))
    assert_size_stride(primals_123, (2240, ), (1, ))
    assert_size_stride(primals_125, (32, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_126, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_127, (224, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_128, (8, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_130, (224, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(primals_132, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_133, (224, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_134, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_135, (224, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_136, (56, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_138, (224, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_140, (224, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_141, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_142, (448, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_143, (56, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_145, (448, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_147, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_148, (448, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_149, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_150, (448, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_151, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_153, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_155, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_156, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_157, (448, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_158, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_160, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_162, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_163, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_164, (448, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_165, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_167, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_169, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_170, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_171, (448, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_172, (112, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_174, (448, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_176, (448, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_177, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_178, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_179, (112, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_181, (896, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_183, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_184, (896, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_185, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_186, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_187, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_189, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_191, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_192, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_193, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_194, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_196, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_198, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_199, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_200, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_201, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_203, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_205, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_206, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_207, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_208, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_210, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_212, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_213, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_214, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_215, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_217, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_219, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_220, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_221, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_222, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_224, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_226, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_227, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_228, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_229, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_231, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_233, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_234, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_235, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_236, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_238, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_240, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_241, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_242, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_243, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_245, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_247, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_248, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_249, (896, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_250, (224, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_252, (896, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_254, (896, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_255, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_256, (2240, 112, 3, 3), (1008, 1, 336, 112))
    assert_size_stride(primals_257, (224, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_259, (2240, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_261, (2240, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_262, (2240, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (32, ), (1, ))
    assert_size_stride(primals_267, (224, ), (1, ))
    assert_size_stride(primals_268, (224, ), (1, ))
    assert_size_stride(primals_269, (224, ), (1, ))
    assert_size_stride(primals_270, (224, ), (1, ))
    assert_size_stride(primals_271, (224, ), (1, ))
    assert_size_stride(primals_272, (224, ), (1, ))
    assert_size_stride(primals_273, (224, ), (1, ))
    assert_size_stride(primals_274, (224, ), (1, ))
    assert_size_stride(primals_275, (224, ), (1, ))
    assert_size_stride(primals_276, (224, ), (1, ))
    assert_size_stride(primals_277, (224, ), (1, ))
    assert_size_stride(primals_278, (224, ), (1, ))
    assert_size_stride(primals_279, (224, ), (1, ))
    assert_size_stride(primals_280, (224, ), (1, ))
    assert_size_stride(primals_281, (448, ), (1, ))
    assert_size_stride(primals_282, (448, ), (1, ))
    assert_size_stride(primals_283, (448, ), (1, ))
    assert_size_stride(primals_284, (448, ), (1, ))
    assert_size_stride(primals_285, (448, ), (1, ))
    assert_size_stride(primals_286, (448, ), (1, ))
    assert_size_stride(primals_287, (448, ), (1, ))
    assert_size_stride(primals_288, (448, ), (1, ))
    assert_size_stride(primals_289, (448, ), (1, ))
    assert_size_stride(primals_290, (448, ), (1, ))
    assert_size_stride(primals_291, (448, ), (1, ))
    assert_size_stride(primals_292, (448, ), (1, ))
    assert_size_stride(primals_293, (448, ), (1, ))
    assert_size_stride(primals_294, (448, ), (1, ))
    assert_size_stride(primals_295, (448, ), (1, ))
    assert_size_stride(primals_296, (448, ), (1, ))
    assert_size_stride(primals_297, (448, ), (1, ))
    assert_size_stride(primals_298, (448, ), (1, ))
    assert_size_stride(primals_299, (448, ), (1, ))
    assert_size_stride(primals_300, (448, ), (1, ))
    assert_size_stride(primals_301, (448, ), (1, ))
    assert_size_stride(primals_302, (448, ), (1, ))
    assert_size_stride(primals_303, (448, ), (1, ))
    assert_size_stride(primals_304, (448, ), (1, ))
    assert_size_stride(primals_305, (448, ), (1, ))
    assert_size_stride(primals_306, (448, ), (1, ))
    assert_size_stride(primals_307, (448, ), (1, ))
    assert_size_stride(primals_308, (448, ), (1, ))
    assert_size_stride(primals_309, (448, ), (1, ))
    assert_size_stride(primals_310, (448, ), (1, ))
    assert_size_stride(primals_311, (448, ), (1, ))
    assert_size_stride(primals_312, (448, ), (1, ))
    assert_size_stride(primals_313, (896, ), (1, ))
    assert_size_stride(primals_314, (896, ), (1, ))
    assert_size_stride(primals_315, (896, ), (1, ))
    assert_size_stride(primals_316, (896, ), (1, ))
    assert_size_stride(primals_317, (896, ), (1, ))
    assert_size_stride(primals_318, (896, ), (1, ))
    assert_size_stride(primals_319, (896, ), (1, ))
    assert_size_stride(primals_320, (896, ), (1, ))
    assert_size_stride(primals_321, (896, ), (1, ))
    assert_size_stride(primals_322, (896, ), (1, ))
    assert_size_stride(primals_323, (896, ), (1, ))
    assert_size_stride(primals_324, (896, ), (1, ))
    assert_size_stride(primals_325, (896, ), (1, ))
    assert_size_stride(primals_326, (896, ), (1, ))
    assert_size_stride(primals_327, (896, ), (1, ))
    assert_size_stride(primals_328, (896, ), (1, ))
    assert_size_stride(primals_329, (896, ), (1, ))
    assert_size_stride(primals_330, (896, ), (1, ))
    assert_size_stride(primals_331, (896, ), (1, ))
    assert_size_stride(primals_332, (896, ), (1, ))
    assert_size_stride(primals_333, (896, ), (1, ))
    assert_size_stride(primals_334, (896, ), (1, ))
    assert_size_stride(primals_335, (896, ), (1, ))
    assert_size_stride(primals_336, (896, ), (1, ))
    assert_size_stride(primals_337, (896, ), (1, ))
    assert_size_stride(primals_338, (896, ), (1, ))
    assert_size_stride(primals_339, (896, ), (1, ))
    assert_size_stride(primals_340, (896, ), (1, ))
    assert_size_stride(primals_341, (896, ), (1, ))
    assert_size_stride(primals_342, (896, ), (1, ))
    assert_size_stride(primals_343, (896, ), (1, ))
    assert_size_stride(primals_344, (896, ), (1, ))
    assert_size_stride(primals_345, (896, ), (1, ))
    assert_size_stride(primals_346, (896, ), (1, ))
    assert_size_stride(primals_347, (896, ), (1, ))
    assert_size_stride(primals_348, (896, ), (1, ))
    assert_size_stride(primals_349, (896, ), (1, ))
    assert_size_stride(primals_350, (896, ), (1, ))
    assert_size_stride(primals_351, (896, ), (1, ))
    assert_size_stride(primals_352, (896, ), (1, ))
    assert_size_stride(primals_353, (896, ), (1, ))
    assert_size_stride(primals_354, (896, ), (1, ))
    assert_size_stride(primals_355, (896, ), (1, ))
    assert_size_stride(primals_356, (896, ), (1, ))
    assert_size_stride(primals_357, (896, ), (1, ))
    assert_size_stride(primals_358, (896, ), (1, ))
    assert_size_stride(primals_359, (896, ), (1, ))
    assert_size_stride(primals_360, (896, ), (1, ))
    assert_size_stride(primals_361, (896, ), (1, ))
    assert_size_stride(primals_362, (896, ), (1, ))
    assert_size_stride(primals_363, (896, ), (1, ))
    assert_size_stride(primals_364, (896, ), (1, ))
    assert_size_stride(primals_365, (896, ), (1, ))
    assert_size_stride(primals_366, (896, ), (1, ))
    assert_size_stride(primals_367, (896, ), (1, ))
    assert_size_stride(primals_368, (896, ), (1, ))
    assert_size_stride(primals_369, (896, ), (1, ))
    assert_size_stride(primals_370, (896, ), (1, ))
    assert_size_stride(primals_371, (896, ), (1, ))
    assert_size_stride(primals_372, (896, ), (1, ))
    assert_size_stride(primals_373, (896, ), (1, ))
    assert_size_stride(primals_374, (896, ), (1, ))
    assert_size_stride(primals_375, (896, ), (1, ))
    assert_size_stride(primals_376, (896, ), (1, ))
    assert_size_stride(primals_377, (896, ), (1, ))
    assert_size_stride(primals_378, (896, ), (1, ))
    assert_size_stride(primals_379, (896, ), (1, ))
    assert_size_stride(primals_380, (896, ), (1, ))
    assert_size_stride(primals_381, (2240, ), (1, ))
    assert_size_stride(primals_382, (2240, ), (1, ))
    assert_size_stride(primals_383, (2240, ), (1, ))
    assert_size_stride(primals_384, (2240, ), (1, ))
    assert_size_stride(primals_385, (2240, ), (1, ))
    assert_size_stride(primals_386, (2240, ), (1, ))
    assert_size_stride(primals_387, (2240, ), (1, ))
    assert_size_stride(primals_388, (2240, ), (1, ))
    assert_size_stride(primals_389, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(relu, (4, 32, 112, 112), (401408, 1, 3584, 32))
    assert_size_stride(convolution_1, (4, 224, 112, 112), (2809856, 1, 25088, 224))
    assert_size_stride(relu_1, (4, 224, 112, 112), (2809856, 1, 25088, 224))
    assert_size_stride(convolution_2, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(relu_2, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(mean, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(relu_3, (4, 8, 1, 1), (8, 1, 8, 8))
    assert_size_stride(convolution_4, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(mul_9, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_5, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_6, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(relu_4, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_7, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(relu_5, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_8, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(relu_6, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(mean_1, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(relu_7, (4, 56, 1, 1), (56, 1, 56, 56))
    assert_size_stride(convolution_10, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(mul_22, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_11, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(relu_8, (4, 224, 56, 56), (702464, 1, 12544, 224))
    assert_size_stride(convolution_12, (4, 448, 56, 56), (1404928, 1, 25088, 448))
    assert_size_stride(relu_9, (4, 448, 56, 56), (1404928, 1, 25088, 448))
    assert_size_stride(convolution_13, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_10, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(mean_2, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(relu_11, (4, 56, 1, 1), (56, 1, 56, 56))
    assert_size_stride(convolution_15, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(mul_32, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_16, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_17, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_12, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_18, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_13, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_19, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_14, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(mean_3, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(relu_15, (4, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(convolution_21, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(mul_45, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_22, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_16, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_23, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_17, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_24, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_18, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(mean_4, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(relu_19, (4, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(convolution_26, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(mul_55, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_27, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_20, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_28, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_21, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_29, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_22, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(mean_5, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(relu_23, (4, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(convolution_31, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(mul_65, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_32, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_24, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_33, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_25, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_34, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_26, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(mean_6, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(relu_27, (4, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(convolution_36, (4, 448, 1, 1), (448, 1, 448, 448))
    assert_size_stride(mul_75, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_37, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(relu_28, (4, 448, 28, 28), (351232, 1, 12544, 448))
    assert_size_stride(convolution_38, (4, 896, 28, 28), (702464, 1, 25088, 896))
    assert_size_stride(relu_29, (4, 896, 28, 28), (702464, 1, 25088, 896))
    assert_size_stride(convolution_39, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_30, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_7, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_31, (4, 112, 1, 1), (112, 1, 112, 112))
    assert_size_stride(convolution_41, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_85, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_42, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_43, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_32, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_44, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_33, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_45, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_34, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_8, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_35, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_47, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_98, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_48, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_36, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_49, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_37, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_50, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_38, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_9, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_39, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_52, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_108, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_53, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_40, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_54, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_41, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_55, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_42, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_10, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_43, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_57, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_118, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_58, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_44, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_59, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_45, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_60, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_46, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_11, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_47, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_62, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_128, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_63, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_48, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_64, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_49, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_65, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_50, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_12, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_51, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_67, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_138, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_68, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_52, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_69, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_53, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_70, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_54, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_13, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_55, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_72, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_148, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_73, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_56, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_74, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_57, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_75, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_58, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_14, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_59, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_77, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_158, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_78, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_60, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_79, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_61, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_80, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_62, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_15, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_63, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_82, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_168, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_83, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_64, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_84, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_65, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_85, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_66, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_16, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_67, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_87, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_178, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_88, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_68, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_89, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_69, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_90, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_70, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(mean_17, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(relu_71, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_92, (4, 896, 1, 1), (896, 1, 896, 896))
    assert_size_stride(mul_188, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_93, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(relu_72, (4, 896, 14, 14), (175616, 1, 12544, 896))
    assert_size_stride(convolution_94, (4, 2240, 14, 14), (439040, 1, 31360, 2240))
    assert_size_stride(relu_73, (4, 2240, 14, 14), (439040, 1, 31360, 2240))
    assert_size_stride(convolution_95, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(relu_74, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(mean_18, (4, 2240, 1, 1), (2240, 1, 2240, 2240))
    assert_size_stride(relu_75, (4, 224, 1, 1), (224, 1, 224, 224))
    assert_size_stride(convolution_97, (4, 2240, 1, 1), (2240, 1, 2240, 2240))
    assert_size_stride(mul_198, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(convolution_98, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(convolution_99, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(clone, (4, 2240), (2240, 1))
    assert_size_stride(permute_1, (1000, 2240), (2240, 1))
    assert_size_stride(le, (4, 2240, 7, 7), (109760, 1, 15680, 2240))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 2240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_1, out=buf0)
    del permute_1
    buf1 = empty((1000, 2240), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), clone, out=buf1)
    del clone
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf4 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf10 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf5 = buf4; del buf4  # reuse
    buf6 = empty_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    buf12 = empty_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_native_batch_norm_backward_sum_threshold_backward_view_0(c_void_p(buf5.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(le.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(convolution_99.data_ptr()), c_void_p(primals_387.data_ptr()), c_void_p(convolution_98.data_ptr()), c_void_p(primals_385.data_ptr()), c_void_p(primals_388.data_ptr()), c_void_p(primals_123.data_ptr()), c_void_p(primals_386.data_ptr()), c_void_p(primals_121.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf10.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf12.data_ptr()))
    del convolution_98
    del convolution_99
    del le
    del primals_121
    del primals_123
    del primals_385
    del primals_387
    del primals_388
    del tangents_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf7 = aten.convolution_backward(buf6, relu_72, primals_262, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf6
    del primals_262
    buf8 = buf7[0]
    buf9 = buf7[1]
    del buf7
    buf11 = buf10; del buf10  # reuse
    cpp_fused_native_batch_norm_backward_1(c_void_p(buf11.data_ptr()), c_void_p(primals_386.data_ptr()))
    del primals_386
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.native_batch_norm_backward, aten.threshold_backward]
    buf13 = aten.convolution_backward(buf12, mul_198, primals_261, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf12
    del mul_198
    del primals_261
    buf14 = buf13[0]
    buf15 = buf13[1]
    del buf13
    buf16 = reinterpret_tensor(buf0, (4, 2240, 1, 1), (2240, 1, 8960, 8960), 0); del buf0  # reuse
    buf17 = reinterpret_tensor(buf16, (4, 2240, 1, 1), (2240, 1, 1, 1), 0); del buf16  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_2(c_void_p(buf17.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(relu_74.data_ptr()), c_void_p(convolution_97.data_ptr()))
    # Source Nodes: [sigmoid_18], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf18 = aten.convolution_backward(buf17, relu_75, primals_259, [2240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf17
    del primals_259
    buf19 = buf18[0]
    buf20 = buf18[1]
    buf21 = buf18[2]
    del buf18
    buf22 = buf19; del buf19  # reuse
    cpp_fused_convolution_backward_threshold_backward_3(c_void_p(buf22.data_ptr()), c_void_p(relu_75.data_ptr()))
    del relu_75
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf23 = aten.convolution_backward(buf22, mean_18, primals_257, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_18
    del primals_257
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf28 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf29 = buf28; del buf28  # reuse
    buf30 = buf14; del buf14  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_4(c_void_p(buf29.data_ptr()), c_void_p(buf30.data_ptr()), c_void_p(relu_74.data_ptr()), c_void_p(convolution_97.data_ptr()), c_void_p(buf24.data_ptr()), c_void_p(convolution_95.data_ptr()), c_void_p(primals_383.data_ptr()), c_void_p(primals_384.data_ptr()), c_void_p(primals_119.data_ptr()), c_void_p(buf27.data_ptr()))
    del buf24
    del convolution_95
    del convolution_97
    del primals_119
    del primals_383
    del primals_384
    del relu_74
    # Source Nodes: [sigmoid_18], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf31 = aten.convolution_backward(buf30, relu_73, primals_256, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 20, [True, True, False])
    del buf30
    del primals_256
    buf32 = buf31[0]
    buf33 = buf31[1]
    del buf31
    buf34 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf35 = empty((2240, ), device='cpu', dtype=torch.float32)
    buf36 = buf35; del buf35  # reuse
    buf37 = buf32; del buf32  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_5(c_void_p(buf36.data_ptr()), c_void_p(buf37.data_ptr()), c_void_p(relu_73.data_ptr()), c_void_p(convolution_94.data_ptr()), c_void_p(primals_381.data_ptr()), c_void_p(primals_382.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf34.data_ptr()))
    del convolution_94
    del primals_117
    del primals_381
    del primals_382
    del relu_73
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf38 = aten.convolution_backward(buf37, relu_72, primals_255, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf37
    del primals_255
    buf39 = buf38[0]
    buf40 = buf38[1]
    del buf38
    buf41 = reinterpret_tensor(buf22, (896, ), (1, ), 0); del buf22  # reuse
    buf42 = empty((896, ), device='cpu', dtype=torch.float32)
    buf43 = buf42; del buf42  # reuse
    buf44 = empty_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_6(c_void_p(buf43.data_ptr()), c_void_p(relu_72.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf39.data_ptr()), c_void_p(convolution_93.data_ptr()), c_void_p(primals_379.data_ptr()), c_void_p(primals_380.data_ptr()), c_void_p(primals_115.data_ptr()), c_void_p(buf41.data_ptr()), c_void_p(buf44.data_ptr()))
    del convolution_93
    del primals_115
    del primals_379
    del primals_380
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf45 = aten.convolution_backward(buf44, mul_188, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf44
    del mul_188
    del primals_254
    buf46 = buf45[0]
    buf47 = buf45[1]
    del buf45
    buf48 = empty_strided((4, 896, 1, 1), (896, 1, 3584, 3584), device='cpu', dtype=torch.float32)
    buf49 = reinterpret_tensor(buf48, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf48  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_7(c_void_p(buf49.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(convolution_92.data_ptr()))
    # Source Nodes: [sigmoid_17], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf50 = aten.convolution_backward(buf49, relu_71, primals_252, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf49
    del primals_252
    buf51 = buf50[0]
    buf52 = buf50[1]
    buf53 = buf50[2]
    del buf50
    buf54 = buf51; del buf51  # reuse
    cpp_fused_convolution_backward_threshold_backward_8(c_void_p(buf54.data_ptr()), c_void_p(relu_71.data_ptr()))
    del relu_71
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf55 = aten.convolution_backward(buf54, mean_17, primals_250, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_17
    del primals_250
    buf56 = buf55[0]
    buf57 = buf55[1]
    buf58 = buf55[2]
    del buf55
    buf59 = reinterpret_tensor(buf54, (896, ), (1, ), 0); del buf54  # reuse
    buf60 = empty((896, ), device='cpu', dtype=torch.float32)
    buf61 = buf60; del buf60  # reuse
    buf62 = buf46; del buf46  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_9(c_void_p(buf61.data_ptr()), c_void_p(buf62.data_ptr()), c_void_p(relu_70.data_ptr()), c_void_p(convolution_92.data_ptr()), c_void_p(buf56.data_ptr()), c_void_p(convolution_90.data_ptr()), c_void_p(primals_377.data_ptr()), c_void_p(primals_378.data_ptr()), c_void_p(primals_113.data_ptr()), c_void_p(buf59.data_ptr()))
    del convolution_90
    del convolution_92
    del primals_113
    del primals_377
    del primals_378
    del relu_70
    # Source Nodes: [sigmoid_17], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf63 = aten.convolution_backward(buf62, relu_69, primals_249, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf62
    del primals_249
    buf64 = buf63[0]
    buf65 = buf63[1]
    del buf63
    buf66 = empty((896, ), device='cpu', dtype=torch.float32)
    buf67 = empty((896, ), device='cpu', dtype=torch.float32)
    buf68 = buf67; del buf67  # reuse
    buf69 = buf64; del buf64  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_10(c_void_p(buf68.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(relu_69.data_ptr()), c_void_p(convolution_89.data_ptr()), c_void_p(primals_375.data_ptr()), c_void_p(primals_376.data_ptr()), c_void_p(primals_111.data_ptr()), c_void_p(buf66.data_ptr()))
    del convolution_89
    del primals_111
    del primals_375
    del primals_376
    del relu_69
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf70 = aten.convolution_backward(buf69, relu_68, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_248
    buf71 = buf70[0]
    buf72 = buf70[1]
    del buf70
    buf73 = buf39; del buf39  # reuse
    buf77 = buf69; del buf69  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_11(c_void_p(buf73.data_ptr()), c_void_p(relu_68.data_ptr()), c_void_p(relu_72.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf71.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(primals_109.data_ptr()), c_void_p(buf77.data_ptr()))
    del buf71
    del buf8
    del primals_109
    del relu_68
    del relu_72
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf78 = aten.convolution_backward(buf77, mul_178, primals_247, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_178
    del primals_247
    buf79 = buf78[0]
    buf81 = reinterpret_tensor(buf56, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf56  # reuse
    buf82 = reinterpret_tensor(buf81, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf81  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_12(c_void_p(buf82.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(relu_66.data_ptr()), c_void_p(convolution_87.data_ptr()))
    # Source Nodes: [sigmoid_16], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf83 = aten.convolution_backward(buf82, relu_67, primals_245, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf82
    del primals_245
    buf84 = buf83[0]
    buf87 = buf84; del buf84  # reuse
    cpp_fused_convolution_backward_threshold_backward_13(c_void_p(buf87.data_ptr()), c_void_p(relu_67.data_ptr()))
    del relu_67
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf88 = aten.convolution_backward(buf87, mean_16, primals_243, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_16
    del primals_243
    buf89 = buf88[0]
    buf95 = buf77; del buf77  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_14(c_void_p(relu_66.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf95.data_ptr()))
    del primals_107
    # Source Nodes: [sigmoid_16], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf96 = aten.convolution_backward(buf95, relu_65, primals_242, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del primals_242
    buf97 = buf96[0]
    buf102 = buf95; del buf95  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_15(c_void_p(relu_65.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(primals_105.data_ptr()), c_void_p(buf102.data_ptr()))
    del primals_105
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf103 = aten.convolution_backward(buf102, relu_64, primals_241, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf102
    del primals_241
    buf104 = buf103[0]
    buf74 = reinterpret_tensor(buf87, (896, ), (1, ), 0); del buf87  # reuse
    buf75 = empty((896, ), device='cpu', dtype=torch.float32)
    buf106 = empty((896, ), device='cpu', dtype=torch.float32)
    buf107 = empty((896, ), device='cpu', dtype=torch.float32)
    buf76 = buf75; del buf75  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_16(c_void_p(buf76.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(convolution_88.data_ptr()), c_void_p(primals_373.data_ptr()), c_void_p(relu_64.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(convolution_83.data_ptr()), c_void_p(primals_367.data_ptr()), c_void_p(primals_374.data_ptr()), c_void_p(buf74.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(buf107.data_ptr()))
    del convolution_83
    del convolution_88
    del primals_367
    del primals_373
    del primals_374
    buf80 = buf78[1]
    del buf78
    buf85 = buf83[1]
    buf86 = buf83[2]
    del buf83
    buf90 = buf88[1]
    buf91 = buf88[2]
    del buf88
    buf92 = empty((896, ), device='cpu', dtype=torch.float32)
    buf93 = empty((896, ), device='cpu', dtype=torch.float32)
    buf94 = buf93; del buf93  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_17(c_void_p(buf94.data_ptr()), c_void_p(relu_66.data_ptr()), c_void_p(buf79.data_ptr()), c_void_p(convolution_87.data_ptr()), c_void_p(buf89.data_ptr()), c_void_p(convolution_85.data_ptr()), c_void_p(primals_371.data_ptr()), c_void_p(primals_372.data_ptr()), c_void_p(buf92.data_ptr()))
    del buf79
    del convolution_85
    del convolution_87
    del primals_371
    del primals_372
    del relu_66
    buf98 = buf96[1]
    del buf96
    buf99 = empty((896, ), device='cpu', dtype=torch.float32)
    buf100 = empty((896, ), device='cpu', dtype=torch.float32)
    buf101 = buf100; del buf100  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_18(c_void_p(buf101.data_ptr()), c_void_p(relu_65.data_ptr()), c_void_p(buf97.data_ptr()), c_void_p(convolution_84.data_ptr()), c_void_p(primals_369.data_ptr()), c_void_p(primals_370.data_ptr()), c_void_p(buf99.data_ptr()))
    del convolution_84
    del primals_369
    del primals_370
    del relu_65
    buf105 = buf103[1]
    del buf103
    buf108 = buf107; del buf107  # reuse
    buf109 = buf97; del buf97  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_19(c_void_p(buf108.data_ptr()), c_void_p(primals_368.data_ptr()), c_void_p(relu_64.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf104.data_ptr()), c_void_p(primals_103.data_ptr()), c_void_p(buf109.data_ptr()))
    del primals_103
    del primals_368
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf110 = aten.convolution_backward(buf109, mul_168, primals_240, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf109
    del mul_168
    del primals_240
    buf111 = buf110[0]
    buf112 = buf110[1]
    del buf110
    buf113 = reinterpret_tensor(buf89, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf89  # reuse
    buf114 = reinterpret_tensor(buf113, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf113  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_20(c_void_p(buf114.data_ptr()), c_void_p(buf111.data_ptr()), c_void_p(relu_62.data_ptr()), c_void_p(convolution_82.data_ptr()))
    # Source Nodes: [sigmoid_15], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf115 = aten.convolution_backward(buf114, relu_63, primals_238, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf114
    del primals_238
    buf116 = buf115[0]
    buf117 = buf115[1]
    buf118 = buf115[2]
    del buf115
    buf119 = buf116; del buf116  # reuse
    cpp_fused_convolution_backward_threshold_backward_21(c_void_p(buf119.data_ptr()), c_void_p(relu_63.data_ptr()))
    del relu_63
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf120 = aten.convolution_backward(buf119, mean_15, primals_236, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_15
    del primals_236
    buf121 = buf120[0]
    buf122 = buf120[1]
    buf123 = buf120[2]
    del buf120
    buf124 = reinterpret_tensor(buf119, (896, ), (1, ), 0); del buf119  # reuse
    buf125 = empty((896, ), device='cpu', dtype=torch.float32)
    buf126 = buf125; del buf125  # reuse
    buf127 = buf111; del buf111  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_22(c_void_p(buf126.data_ptr()), c_void_p(buf127.data_ptr()), c_void_p(relu_62.data_ptr()), c_void_p(convolution_82.data_ptr()), c_void_p(buf121.data_ptr()), c_void_p(convolution_80.data_ptr()), c_void_p(primals_365.data_ptr()), c_void_p(primals_366.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf124.data_ptr()))
    del convolution_80
    del convolution_82
    del primals_101
    del primals_365
    del primals_366
    del relu_62
    # Source Nodes: [sigmoid_15], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf128 = aten.convolution_backward(buf127, relu_61, primals_235, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf127
    del primals_235
    buf129 = buf128[0]
    buf130 = buf128[1]
    del buf128
    buf131 = empty((896, ), device='cpu', dtype=torch.float32)
    buf132 = empty((896, ), device='cpu', dtype=torch.float32)
    buf133 = buf132; del buf132  # reuse
    buf134 = buf129; del buf129  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_23(c_void_p(buf133.data_ptr()), c_void_p(buf134.data_ptr()), c_void_p(relu_61.data_ptr()), c_void_p(convolution_79.data_ptr()), c_void_p(primals_363.data_ptr()), c_void_p(primals_364.data_ptr()), c_void_p(primals_99.data_ptr()), c_void_p(buf131.data_ptr()))
    del convolution_79
    del primals_363
    del primals_364
    del primals_99
    del relu_61
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf135 = aten.convolution_backward(buf134, relu_60, primals_234, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_234
    buf136 = buf135[0]
    buf137 = buf135[1]
    del buf135
    buf138 = buf104; del buf104  # reuse
    buf142 = buf134; del buf134  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_24(c_void_p(buf138.data_ptr()), c_void_p(relu_60.data_ptr()), c_void_p(relu_64.data_ptr()), c_void_p(buf73.data_ptr()), c_void_p(buf136.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(primals_97.data_ptr()), c_void_p(buf142.data_ptr()))
    del buf136
    del buf73
    del primals_97
    del relu_60
    del relu_64
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf143 = aten.convolution_backward(buf142, mul_158, primals_233, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_158
    del primals_233
    buf144 = buf143[0]
    buf146 = reinterpret_tensor(buf121, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf121  # reuse
    buf147 = reinterpret_tensor(buf146, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf146  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_25(c_void_p(buf147.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(relu_58.data_ptr()), c_void_p(convolution_77.data_ptr()))
    # Source Nodes: [sigmoid_14], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf148 = aten.convolution_backward(buf147, relu_59, primals_231, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf147
    del primals_231
    buf149 = buf148[0]
    buf152 = buf149; del buf149  # reuse
    cpp_fused_convolution_backward_threshold_backward_26(c_void_p(buf152.data_ptr()), c_void_p(relu_59.data_ptr()))
    del relu_59
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf153 = aten.convolution_backward(buf152, mean_14, primals_229, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_14
    del primals_229
    buf154 = buf153[0]
    buf160 = buf142; del buf142  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_27(c_void_p(relu_58.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf160.data_ptr()))
    del primals_95
    # Source Nodes: [sigmoid_14], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf161 = aten.convolution_backward(buf160, relu_57, primals_228, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del primals_228
    buf162 = buf161[0]
    buf167 = buf160; del buf160  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_28(c_void_p(relu_57.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(primals_93.data_ptr()), c_void_p(buf167.data_ptr()))
    del primals_93
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf168 = aten.convolution_backward(buf167, relu_56, primals_227, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf167
    del primals_227
    buf169 = buf168[0]
    buf139 = reinterpret_tensor(buf152, (896, ), (1, ), 0); del buf152  # reuse
    buf140 = empty((896, ), device='cpu', dtype=torch.float32)
    buf171 = empty((896, ), device='cpu', dtype=torch.float32)
    buf172 = empty((896, ), device='cpu', dtype=torch.float32)
    buf141 = buf140; del buf140  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_29(c_void_p(buf141.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(convolution_78.data_ptr()), c_void_p(primals_361.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(convolution_73.data_ptr()), c_void_p(primals_355.data_ptr()), c_void_p(primals_362.data_ptr()), c_void_p(buf139.data_ptr()), c_void_p(buf171.data_ptr()), c_void_p(buf172.data_ptr()))
    del convolution_73
    del convolution_78
    del primals_355
    del primals_361
    del primals_362
    buf145 = buf143[1]
    del buf143
    buf150 = buf148[1]
    buf151 = buf148[2]
    del buf148
    buf155 = buf153[1]
    buf156 = buf153[2]
    del buf153
    buf157 = empty((896, ), device='cpu', dtype=torch.float32)
    buf158 = empty((896, ), device='cpu', dtype=torch.float32)
    buf159 = buf158; del buf158  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_30(c_void_p(buf159.data_ptr()), c_void_p(relu_58.data_ptr()), c_void_p(buf144.data_ptr()), c_void_p(convolution_77.data_ptr()), c_void_p(buf154.data_ptr()), c_void_p(convolution_75.data_ptr()), c_void_p(primals_359.data_ptr()), c_void_p(primals_360.data_ptr()), c_void_p(buf157.data_ptr()))
    del buf144
    del convolution_75
    del convolution_77
    del primals_359
    del primals_360
    del relu_58
    buf163 = buf161[1]
    del buf161
    buf164 = empty((896, ), device='cpu', dtype=torch.float32)
    buf165 = empty((896, ), device='cpu', dtype=torch.float32)
    buf166 = buf165; del buf165  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_31(c_void_p(buf166.data_ptr()), c_void_p(relu_57.data_ptr()), c_void_p(buf162.data_ptr()), c_void_p(convolution_74.data_ptr()), c_void_p(primals_357.data_ptr()), c_void_p(primals_358.data_ptr()), c_void_p(buf164.data_ptr()))
    del convolution_74
    del primals_357
    del primals_358
    del relu_57
    buf170 = buf168[1]
    del buf168
    buf173 = buf172; del buf172  # reuse
    buf174 = buf162; del buf162  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_32(c_void_p(buf173.data_ptr()), c_void_p(primals_356.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(buf138.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(primals_91.data_ptr()), c_void_p(buf174.data_ptr()))
    del primals_356
    del primals_91
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf175 = aten.convolution_backward(buf174, mul_148, primals_226, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf174
    del mul_148
    del primals_226
    buf176 = buf175[0]
    buf177 = buf175[1]
    del buf175
    buf178 = reinterpret_tensor(buf154, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf154  # reuse
    buf179 = reinterpret_tensor(buf178, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf178  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_33(c_void_p(buf179.data_ptr()), c_void_p(buf176.data_ptr()), c_void_p(relu_54.data_ptr()), c_void_p(convolution_72.data_ptr()))
    # Source Nodes: [sigmoid_13], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf180 = aten.convolution_backward(buf179, relu_55, primals_224, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf179
    del primals_224
    buf181 = buf180[0]
    buf182 = buf180[1]
    buf183 = buf180[2]
    del buf180
    buf184 = buf181; del buf181  # reuse
    cpp_fused_convolution_backward_threshold_backward_34(c_void_p(buf184.data_ptr()), c_void_p(relu_55.data_ptr()))
    del relu_55
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf185 = aten.convolution_backward(buf184, mean_13, primals_222, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_13
    del primals_222
    buf186 = buf185[0]
    buf187 = buf185[1]
    buf188 = buf185[2]
    del buf185
    buf189 = reinterpret_tensor(buf184, (896, ), (1, ), 0); del buf184  # reuse
    buf190 = empty((896, ), device='cpu', dtype=torch.float32)
    buf191 = buf190; del buf190  # reuse
    buf192 = buf176; del buf176  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_35(c_void_p(buf191.data_ptr()), c_void_p(buf192.data_ptr()), c_void_p(relu_54.data_ptr()), c_void_p(convolution_72.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(convolution_70.data_ptr()), c_void_p(primals_353.data_ptr()), c_void_p(primals_354.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf189.data_ptr()))
    del convolution_70
    del convolution_72
    del primals_353
    del primals_354
    del primals_89
    del relu_54
    # Source Nodes: [sigmoid_13], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf193 = aten.convolution_backward(buf192, relu_53, primals_221, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf192
    del primals_221
    buf194 = buf193[0]
    buf195 = buf193[1]
    del buf193
    buf196 = empty((896, ), device='cpu', dtype=torch.float32)
    buf197 = empty((896, ), device='cpu', dtype=torch.float32)
    buf198 = buf197; del buf197  # reuse
    buf199 = buf194; del buf194  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_36(c_void_p(buf198.data_ptr()), c_void_p(buf199.data_ptr()), c_void_p(relu_53.data_ptr()), c_void_p(convolution_69.data_ptr()), c_void_p(primals_351.data_ptr()), c_void_p(primals_352.data_ptr()), c_void_p(primals_87.data_ptr()), c_void_p(buf196.data_ptr()))
    del convolution_69
    del primals_351
    del primals_352
    del primals_87
    del relu_53
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf200 = aten.convolution_backward(buf199, relu_52, primals_220, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_220
    buf201 = buf200[0]
    buf202 = buf200[1]
    del buf200
    buf203 = buf138; del buf138  # reuse
    buf207 = buf199; del buf199  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_37(c_void_p(buf203.data_ptr()), c_void_p(relu_52.data_ptr()), c_void_p(relu_56.data_ptr()), c_void_p(buf169.data_ptr()), c_void_p(buf201.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(primals_85.data_ptr()), c_void_p(buf207.data_ptr()))
    del buf169
    del buf201
    del primals_85
    del relu_52
    del relu_56
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf208 = aten.convolution_backward(buf207, mul_138, primals_219, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_138
    del primals_219
    buf209 = buf208[0]
    buf211 = reinterpret_tensor(buf186, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf186  # reuse
    buf212 = reinterpret_tensor(buf211, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf211  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_38(c_void_p(buf212.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(convolution_67.data_ptr()))
    # Source Nodes: [sigmoid_12], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf213 = aten.convolution_backward(buf212, relu_51, primals_217, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf212
    del primals_217
    buf214 = buf213[0]
    buf217 = buf214; del buf214  # reuse
    cpp_fused_convolution_backward_threshold_backward_39(c_void_p(buf217.data_ptr()), c_void_p(relu_51.data_ptr()))
    del relu_51
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf218 = aten.convolution_backward(buf217, mean_12, primals_215, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_12
    del primals_215
    buf219 = buf218[0]
    buf225 = buf207; del buf207  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_40(c_void_p(relu_50.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf225.data_ptr()))
    del primals_83
    # Source Nodes: [sigmoid_12], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf226 = aten.convolution_backward(buf225, relu_49, primals_214, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del primals_214
    buf227 = buf226[0]
    buf232 = buf225; del buf225  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_41(c_void_p(relu_49.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(primals_81.data_ptr()), c_void_p(buf232.data_ptr()))
    del primals_81
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf233 = aten.convolution_backward(buf232, relu_48, primals_213, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf232
    del primals_213
    buf234 = buf233[0]
    buf204 = reinterpret_tensor(buf217, (896, ), (1, ), 0); del buf217  # reuse
    buf205 = empty((896, ), device='cpu', dtype=torch.float32)
    buf236 = empty((896, ), device='cpu', dtype=torch.float32)
    buf237 = empty((896, ), device='cpu', dtype=torch.float32)
    buf206 = buf205; del buf205  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_42(c_void_p(buf206.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(convolution_68.data_ptr()), c_void_p(primals_349.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(convolution_63.data_ptr()), c_void_p(primals_343.data_ptr()), c_void_p(primals_350.data_ptr()), c_void_p(buf204.data_ptr()), c_void_p(buf236.data_ptr()), c_void_p(buf237.data_ptr()))
    del convolution_63
    del convolution_68
    del primals_343
    del primals_349
    del primals_350
    buf210 = buf208[1]
    del buf208
    buf215 = buf213[1]
    buf216 = buf213[2]
    del buf213
    buf220 = buf218[1]
    buf221 = buf218[2]
    del buf218
    buf222 = empty((896, ), device='cpu', dtype=torch.float32)
    buf223 = empty((896, ), device='cpu', dtype=torch.float32)
    buf224 = buf223; del buf223  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_43(c_void_p(buf224.data_ptr()), c_void_p(relu_50.data_ptr()), c_void_p(buf209.data_ptr()), c_void_p(convolution_67.data_ptr()), c_void_p(buf219.data_ptr()), c_void_p(convolution_65.data_ptr()), c_void_p(primals_347.data_ptr()), c_void_p(primals_348.data_ptr()), c_void_p(buf222.data_ptr()))
    del buf209
    del convolution_65
    del convolution_67
    del primals_347
    del primals_348
    del relu_50
    buf228 = buf226[1]
    del buf226
    buf229 = empty((896, ), device='cpu', dtype=torch.float32)
    buf230 = empty((896, ), device='cpu', dtype=torch.float32)
    buf231 = buf230; del buf230  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_44(c_void_p(buf231.data_ptr()), c_void_p(relu_49.data_ptr()), c_void_p(buf227.data_ptr()), c_void_p(convolution_64.data_ptr()), c_void_p(primals_345.data_ptr()), c_void_p(primals_346.data_ptr()), c_void_p(buf229.data_ptr()))
    del convolution_64
    del primals_345
    del primals_346
    del relu_49
    buf235 = buf233[1]
    del buf233
    buf238 = buf237; del buf237  # reuse
    buf239 = buf227; del buf227  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_45(c_void_p(buf238.data_ptr()), c_void_p(primals_344.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(buf203.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(primals_79.data_ptr()), c_void_p(buf239.data_ptr()))
    del primals_344
    del primals_79
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf240 = aten.convolution_backward(buf239, mul_128, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf239
    del mul_128
    del primals_212
    buf241 = buf240[0]
    buf242 = buf240[1]
    del buf240
    buf243 = reinterpret_tensor(buf219, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf219  # reuse
    buf244 = reinterpret_tensor(buf243, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf243  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_46(c_void_p(buf244.data_ptr()), c_void_p(buf241.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_62.data_ptr()))
    # Source Nodes: [sigmoid_11], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf245 = aten.convolution_backward(buf244, relu_47, primals_210, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf244
    del primals_210
    buf246 = buf245[0]
    buf247 = buf245[1]
    buf248 = buf245[2]
    del buf245
    buf249 = buf246; del buf246  # reuse
    cpp_fused_convolution_backward_threshold_backward_47(c_void_p(buf249.data_ptr()), c_void_p(relu_47.data_ptr()))
    del relu_47
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf250 = aten.convolution_backward(buf249, mean_11, primals_208, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_11
    del primals_208
    buf251 = buf250[0]
    buf252 = buf250[1]
    buf253 = buf250[2]
    del buf250
    buf254 = reinterpret_tensor(buf249, (896, ), (1, ), 0); del buf249  # reuse
    buf255 = empty((896, ), device='cpu', dtype=torch.float32)
    buf256 = buf255; del buf255  # reuse
    buf257 = buf241; del buf241  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_48(c_void_p(buf256.data_ptr()), c_void_p(buf257.data_ptr()), c_void_p(relu_46.data_ptr()), c_void_p(convolution_62.data_ptr()), c_void_p(buf251.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(primals_341.data_ptr()), c_void_p(primals_342.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf254.data_ptr()))
    del convolution_60
    del convolution_62
    del primals_341
    del primals_342
    del primals_77
    del relu_46
    # Source Nodes: [sigmoid_11], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf258 = aten.convolution_backward(buf257, relu_45, primals_207, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf257
    del primals_207
    buf259 = buf258[0]
    buf260 = buf258[1]
    del buf258
    buf261 = empty((896, ), device='cpu', dtype=torch.float32)
    buf262 = empty((896, ), device='cpu', dtype=torch.float32)
    buf263 = buf262; del buf262  # reuse
    buf264 = buf259; del buf259  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_49(c_void_p(buf263.data_ptr()), c_void_p(buf264.data_ptr()), c_void_p(relu_45.data_ptr()), c_void_p(convolution_59.data_ptr()), c_void_p(primals_339.data_ptr()), c_void_p(primals_340.data_ptr()), c_void_p(primals_75.data_ptr()), c_void_p(buf261.data_ptr()))
    del convolution_59
    del primals_339
    del primals_340
    del primals_75
    del relu_45
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf265 = aten.convolution_backward(buf264, relu_44, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_206
    buf266 = buf265[0]
    buf267 = buf265[1]
    del buf265
    buf268 = buf203; del buf203  # reuse
    buf272 = buf264; del buf264  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_50(c_void_p(buf268.data_ptr()), c_void_p(relu_44.data_ptr()), c_void_p(relu_48.data_ptr()), c_void_p(buf234.data_ptr()), c_void_p(buf266.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(primals_73.data_ptr()), c_void_p(buf272.data_ptr()))
    del buf234
    del buf266
    del primals_73
    del relu_44
    del relu_48
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf273 = aten.convolution_backward(buf272, mul_118, primals_205, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_118
    del primals_205
    buf274 = buf273[0]
    buf276 = reinterpret_tensor(buf251, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf251  # reuse
    buf277 = reinterpret_tensor(buf276, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf276  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_51(c_void_p(buf277.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(convolution_57.data_ptr()))
    # Source Nodes: [sigmoid_10], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf278 = aten.convolution_backward(buf277, relu_43, primals_203, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf277
    del primals_203
    buf279 = buf278[0]
    buf282 = buf279; del buf279  # reuse
    cpp_fused_convolution_backward_threshold_backward_52(c_void_p(buf282.data_ptr()), c_void_p(relu_43.data_ptr()))
    del relu_43
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf283 = aten.convolution_backward(buf282, mean_10, primals_201, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_10
    del primals_201
    buf284 = buf283[0]
    buf290 = buf272; del buf272  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_53(c_void_p(relu_42.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf290.data_ptr()))
    del primals_71
    # Source Nodes: [sigmoid_10], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf291 = aten.convolution_backward(buf290, relu_41, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del primals_200
    buf292 = buf291[0]
    buf297 = buf290; del buf290  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_54(c_void_p(relu_41.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(primals_69.data_ptr()), c_void_p(buf297.data_ptr()))
    del primals_69
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf298 = aten.convolution_backward(buf297, relu_40, primals_199, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf297
    del primals_199
    buf299 = buf298[0]
    buf269 = reinterpret_tensor(buf282, (896, ), (1, ), 0); del buf282  # reuse
    buf270 = empty((896, ), device='cpu', dtype=torch.float32)
    buf301 = empty((896, ), device='cpu', dtype=torch.float32)
    buf302 = empty((896, ), device='cpu', dtype=torch.float32)
    buf271 = buf270; del buf270  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_55(c_void_p(buf271.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(convolution_58.data_ptr()), c_void_p(primals_337.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(convolution_53.data_ptr()), c_void_p(primals_331.data_ptr()), c_void_p(primals_338.data_ptr()), c_void_p(buf269.data_ptr()), c_void_p(buf301.data_ptr()), c_void_p(buf302.data_ptr()))
    del convolution_53
    del convolution_58
    del primals_331
    del primals_337
    del primals_338
    buf275 = buf273[1]
    del buf273
    buf280 = buf278[1]
    buf281 = buf278[2]
    del buf278
    buf285 = buf283[1]
    buf286 = buf283[2]
    del buf283
    buf287 = empty((896, ), device='cpu', dtype=torch.float32)
    buf288 = empty((896, ), device='cpu', dtype=torch.float32)
    buf289 = buf288; del buf288  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_56(c_void_p(buf289.data_ptr()), c_void_p(relu_42.data_ptr()), c_void_p(buf274.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(buf284.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(primals_335.data_ptr()), c_void_p(primals_336.data_ptr()), c_void_p(buf287.data_ptr()))
    del buf274
    del convolution_55
    del convolution_57
    del primals_335
    del primals_336
    del relu_42
    buf293 = buf291[1]
    del buf291
    buf294 = empty((896, ), device='cpu', dtype=torch.float32)
    buf295 = empty((896, ), device='cpu', dtype=torch.float32)
    buf296 = buf295; del buf295  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_57(c_void_p(buf296.data_ptr()), c_void_p(relu_41.data_ptr()), c_void_p(buf292.data_ptr()), c_void_p(convolution_54.data_ptr()), c_void_p(primals_333.data_ptr()), c_void_p(primals_334.data_ptr()), c_void_p(buf294.data_ptr()))
    del convolution_54
    del primals_333
    del primals_334
    del relu_41
    buf300 = buf298[1]
    del buf298
    buf303 = buf302; del buf302  # reuse
    buf304 = buf292; del buf292  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf303.data_ptr()), c_void_p(primals_332.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf268.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(primals_67.data_ptr()), c_void_p(buf304.data_ptr()))
    del primals_332
    del primals_67
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf305 = aten.convolution_backward(buf304, mul_108, primals_198, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf304
    del mul_108
    del primals_198
    buf306 = buf305[0]
    buf307 = buf305[1]
    del buf305
    buf308 = reinterpret_tensor(buf284, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf284  # reuse
    buf309 = reinterpret_tensor(buf308, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf308  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_59(c_void_p(buf309.data_ptr()), c_void_p(buf306.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_52.data_ptr()))
    # Source Nodes: [sigmoid_9], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf310 = aten.convolution_backward(buf309, relu_39, primals_196, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf309
    del primals_196
    buf311 = buf310[0]
    buf312 = buf310[1]
    buf313 = buf310[2]
    del buf310
    buf314 = buf311; del buf311  # reuse
    cpp_fused_convolution_backward_threshold_backward_60(c_void_p(buf314.data_ptr()), c_void_p(relu_39.data_ptr()))
    del relu_39
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf315 = aten.convolution_backward(buf314, mean_9, primals_194, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_9
    del primals_194
    buf316 = buf315[0]
    buf317 = buf315[1]
    buf318 = buf315[2]
    del buf315
    buf319 = reinterpret_tensor(buf314, (896, ), (1, ), 0); del buf314  # reuse
    buf320 = empty((896, ), device='cpu', dtype=torch.float32)
    buf321 = buf320; del buf320  # reuse
    buf322 = buf306; del buf306  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_61(c_void_p(buf321.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(relu_38.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(buf316.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_329.data_ptr()), c_void_p(primals_330.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf319.data_ptr()))
    del convolution_50
    del convolution_52
    del primals_329
    del primals_330
    del primals_65
    del relu_38
    # Source Nodes: [sigmoid_9], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf323 = aten.convolution_backward(buf322, relu_37, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf322
    del primals_193
    buf324 = buf323[0]
    buf325 = buf323[1]
    del buf323
    buf326 = empty((896, ), device='cpu', dtype=torch.float32)
    buf327 = empty((896, ), device='cpu', dtype=torch.float32)
    buf328 = buf327; del buf327  # reuse
    buf329 = buf324; del buf324  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_62(c_void_p(buf328.data_ptr()), c_void_p(buf329.data_ptr()), c_void_p(relu_37.data_ptr()), c_void_p(convolution_49.data_ptr()), c_void_p(primals_327.data_ptr()), c_void_p(primals_328.data_ptr()), c_void_p(primals_63.data_ptr()), c_void_p(buf326.data_ptr()))
    del convolution_49
    del primals_327
    del primals_328
    del primals_63
    del relu_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf330 = aten.convolution_backward(buf329, relu_36, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_192
    buf331 = buf330[0]
    buf332 = buf330[1]
    del buf330
    buf333 = buf268; del buf268  # reuse
    buf337 = buf329; del buf329  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(buf333.data_ptr()), c_void_p(relu_36.data_ptr()), c_void_p(relu_40.data_ptr()), c_void_p(buf299.data_ptr()), c_void_p(buf331.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(primals_61.data_ptr()), c_void_p(buf337.data_ptr()))
    del buf299
    del buf331
    del primals_61
    del relu_36
    del relu_40
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf338 = aten.convolution_backward(buf337, mul_98, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_98
    del primals_191
    buf339 = buf338[0]
    buf341 = reinterpret_tensor(buf316, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf316  # reuse
    buf342 = reinterpret_tensor(buf341, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf341  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_64(c_void_p(buf342.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(convolution_47.data_ptr()))
    # Source Nodes: [sigmoid_8], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf343 = aten.convolution_backward(buf342, relu_35, primals_189, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf342
    del primals_189
    buf344 = buf343[0]
    buf347 = buf344; del buf344  # reuse
    cpp_fused_convolution_backward_threshold_backward_65(c_void_p(buf347.data_ptr()), c_void_p(relu_35.data_ptr()))
    del relu_35
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf348 = aten.convolution_backward(buf347, mean_8, primals_187, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_8
    del primals_187
    buf349 = buf348[0]
    buf355 = buf337; del buf337  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_66(c_void_p(relu_34.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(primals_59.data_ptr()), c_void_p(buf355.data_ptr()))
    del primals_59
    # Source Nodes: [sigmoid_8], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf356 = aten.convolution_backward(buf355, relu_33, primals_186, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del primals_186
    buf357 = buf356[0]
    buf362 = buf355; del buf355  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_67(c_void_p(relu_33.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(primals_57.data_ptr()), c_void_p(buf362.data_ptr()))
    del primals_57
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf363 = aten.convolution_backward(buf362, relu_32, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf362
    del primals_185
    buf364 = buf363[0]
    buf334 = reinterpret_tensor(buf347, (896, ), (1, ), 0); del buf347  # reuse
    buf335 = empty((896, ), device='cpu', dtype=torch.float32)
    buf366 = empty((896, ), device='cpu', dtype=torch.float32)
    buf367 = empty((896, ), device='cpu', dtype=torch.float32)
    buf373 = empty((896, ), device='cpu', dtype=torch.float32)
    buf336 = buf335; del buf335  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_68(c_void_p(buf336.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(convolution_48.data_ptr()), c_void_p(primals_325.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(convolution_43.data_ptr()), c_void_p(primals_319.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_317.data_ptr()), c_void_p(primals_326.data_ptr()), c_void_p(buf334.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(buf367.data_ptr()), c_void_p(buf373.data_ptr()))
    del convolution_42
    del convolution_43
    del convolution_48
    del primals_317
    del primals_319
    del primals_325
    del primals_326
    buf340 = buf338[1]
    del buf338
    buf345 = buf343[1]
    buf346 = buf343[2]
    del buf343
    buf350 = buf348[1]
    buf351 = buf348[2]
    del buf348
    buf352 = empty((896, ), device='cpu', dtype=torch.float32)
    buf353 = empty((896, ), device='cpu', dtype=torch.float32)
    buf354 = buf353; del buf353  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_69(c_void_p(buf354.data_ptr()), c_void_p(relu_34.data_ptr()), c_void_p(buf339.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(buf349.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_323.data_ptr()), c_void_p(primals_324.data_ptr()), c_void_p(buf352.data_ptr()))
    del convolution_45
    del convolution_47
    del primals_323
    del primals_324
    del relu_34
    buf358 = buf356[1]
    del buf356
    buf359 = empty((896, ), device='cpu', dtype=torch.float32)
    buf360 = empty((896, ), device='cpu', dtype=torch.float32)
    buf361 = buf360; del buf360  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_70(c_void_p(buf361.data_ptr()), c_void_p(relu_33.data_ptr()), c_void_p(buf357.data_ptr()), c_void_p(convolution_44.data_ptr()), c_void_p(primals_321.data_ptr()), c_void_p(primals_322.data_ptr()), c_void_p(buf359.data_ptr()))
    del convolution_44
    del primals_321
    del primals_322
    del relu_33
    buf365 = buf363[1]
    del buf363
    buf368 = buf367; del buf367  # reuse
    buf369 = buf357; del buf357  # reuse
    buf375 = buf339; del buf339  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_71(c_void_p(buf368.data_ptr()), c_void_p(primals_320.data_ptr()), c_void_p(relu_32.data_ptr()), c_void_p(buf333.data_ptr()), c_void_p(buf364.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(primals_318.data_ptr()), c_void_p(primals_53.data_ptr()), c_void_p(buf369.data_ptr()), c_void_p(buf375.data_ptr()))
    del buf333
    del buf364
    del primals_320
    del primals_53
    del primals_55
    del relu_32
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf370 = aten.convolution_backward(buf369, relu_28, primals_184, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf369
    del primals_184
    buf371 = buf370[0]
    buf372 = buf370[1]
    del buf370
    buf374 = buf373; del buf373  # reuse
    cpp_fused_native_batch_norm_backward_72(c_void_p(buf374.data_ptr()), c_void_p(primals_318.data_ptr()))
    del primals_318
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf376 = aten.convolution_backward(buf375, mul_85, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf375
    del mul_85
    del primals_183
    buf377 = buf376[0]
    buf378 = buf376[1]
    del buf376
    buf379 = reinterpret_tensor(buf349, (4, 896, 1, 1), (896, 1, 3584, 3584), 0); del buf349  # reuse
    buf380 = reinterpret_tensor(buf379, (4, 896, 1, 1), (896, 1, 1, 1), 0); del buf379  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_73(c_void_p(buf380.data_ptr()), c_void_p(buf377.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(convolution_41.data_ptr()))
    # Source Nodes: [sigmoid_7], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf381 = aten.convolution_backward(buf380, relu_31, primals_181, [896], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf380
    del primals_181
    buf382 = buf381[0]
    buf383 = buf381[1]
    buf384 = buf381[2]
    del buf381
    buf385 = buf382; del buf382  # reuse
    cpp_fused_convolution_backward_threshold_backward_74(c_void_p(buf385.data_ptr()), c_void_p(relu_31.data_ptr()))
    del relu_31
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf386 = aten.convolution_backward(buf385, mean_7, primals_179, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_7
    del primals_179
    buf387 = buf386[0]
    buf388 = buf386[1]
    buf389 = buf386[2]
    del buf386
    buf390 = empty((896, ), device='cpu', dtype=torch.float32)
    buf391 = empty((896, ), device='cpu', dtype=torch.float32)
    buf392 = buf391; del buf391  # reuse
    buf393 = buf377; del buf377  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_75(c_void_p(buf392.data_ptr()), c_void_p(buf393.data_ptr()), c_void_p(relu_30.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(buf387.data_ptr()), c_void_p(convolution_39.data_ptr()), c_void_p(primals_315.data_ptr()), c_void_p(primals_316.data_ptr()), c_void_p(primals_51.data_ptr()), c_void_p(buf390.data_ptr()))
    del buf387
    del convolution_39
    del convolution_41
    del primals_315
    del primals_316
    del primals_51
    del relu_30
    # Source Nodes: [sigmoid_7], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf394 = aten.convolution_backward(buf393, relu_29, primals_178, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 8, [True, True, False])
    del buf393
    del primals_178
    buf395 = buf394[0]
    buf396 = buf394[1]
    del buf394
    buf397 = empty((896, ), device='cpu', dtype=torch.float32)
    buf398 = empty((896, ), device='cpu', dtype=torch.float32)
    buf399 = buf398; del buf398  # reuse
    buf400 = buf395; del buf395  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_76(c_void_p(buf399.data_ptr()), c_void_p(buf400.data_ptr()), c_void_p(relu_29.data_ptr()), c_void_p(convolution_38.data_ptr()), c_void_p(primals_313.data_ptr()), c_void_p(primals_314.data_ptr()), c_void_p(primals_49.data_ptr()), c_void_p(buf397.data_ptr()))
    del convolution_38
    del primals_313
    del primals_314
    del primals_49
    del relu_29
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf401 = aten.convolution_backward(buf400, relu_28, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_177
    buf402 = buf401[0]
    buf403 = buf401[1]
    del buf401
    buf404 = reinterpret_tensor(buf385, (448, ), (1, ), 0); del buf385  # reuse
    buf405 = empty((448, ), device='cpu', dtype=torch.float32)
    buf406 = buf405; del buf405  # reuse
    buf407 = empty_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf406.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(buf371.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_312.data_ptr()), c_void_p(primals_47.data_ptr()), c_void_p(buf404.data_ptr()), c_void_p(buf407.data_ptr()))
    del convolution_37
    del primals_311
    del primals_312
    del primals_47
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf408 = aten.convolution_backward(buf407, mul_75, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf407
    del mul_75
    del primals_176
    buf409 = buf408[0]
    buf410 = buf408[1]
    del buf408
    buf411 = empty_strided((4, 448, 1, 1), (448, 1, 1792, 1792), device='cpu', dtype=torch.float32)
    buf412 = reinterpret_tensor(buf411, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf411  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_78(c_void_p(buf412.data_ptr()), c_void_p(buf409.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_36.data_ptr()))
    # Source Nodes: [sigmoid_6], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf413 = aten.convolution_backward(buf412, relu_27, primals_174, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf412
    del primals_174
    buf414 = buf413[0]
    buf415 = buf413[1]
    buf416 = buf413[2]
    del buf413
    buf417 = buf414; del buf414  # reuse
    cpp_fused_convolution_backward_threshold_backward_79(c_void_p(buf417.data_ptr()), c_void_p(relu_27.data_ptr()))
    del relu_27
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf418 = aten.convolution_backward(buf417, mean_6, primals_172, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_6
    del primals_172
    buf419 = buf418[0]
    buf420 = buf418[1]
    buf421 = buf418[2]
    del buf418
    buf422 = reinterpret_tensor(buf417, (448, ), (1, ), 0); del buf417  # reuse
    buf423 = empty((448, ), device='cpu', dtype=torch.float32)
    buf424 = buf423; del buf423  # reuse
    buf425 = buf409; del buf409  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_80(c_void_p(buf424.data_ptr()), c_void_p(buf425.data_ptr()), c_void_p(relu_26.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(buf419.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_309.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf422.data_ptr()))
    del convolution_34
    del convolution_36
    del primals_309
    del primals_310
    del primals_45
    del relu_26
    # Source Nodes: [sigmoid_6], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf426 = aten.convolution_backward(buf425, relu_25, primals_171, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf425
    del primals_171
    buf427 = buf426[0]
    buf428 = buf426[1]
    del buf426
    buf429 = empty((448, ), device='cpu', dtype=torch.float32)
    buf430 = empty((448, ), device='cpu', dtype=torch.float32)
    buf431 = buf430; del buf430  # reuse
    buf432 = buf427; del buf427  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_81(c_void_p(buf431.data_ptr()), c_void_p(buf432.data_ptr()), c_void_p(relu_25.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_43.data_ptr()), c_void_p(buf429.data_ptr()))
    del convolution_33
    del primals_307
    del primals_308
    del primals_43
    del relu_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf433 = aten.convolution_backward(buf432, relu_24, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_170
    buf434 = buf433[0]
    buf435 = buf433[1]
    del buf433
    buf436 = buf371; del buf371  # reuse
    buf440 = buf432; del buf432  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_82(c_void_p(buf436.data_ptr()), c_void_p(relu_24.data_ptr()), c_void_p(relu_28.data_ptr()), c_void_p(buf402.data_ptr()), c_void_p(buf434.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(primals_41.data_ptr()), c_void_p(buf440.data_ptr()))
    del buf402
    del buf434
    del primals_41
    del relu_24
    del relu_28
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf441 = aten.convolution_backward(buf440, mul_65, primals_169, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_65
    del primals_169
    buf442 = buf441[0]
    buf444 = reinterpret_tensor(buf419, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf419  # reuse
    buf445 = reinterpret_tensor(buf444, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf444  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_83(c_void_p(buf445.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(convolution_31.data_ptr()))
    # Source Nodes: [sigmoid_5], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf446 = aten.convolution_backward(buf445, relu_23, primals_167, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf445
    del primals_167
    buf447 = buf446[0]
    buf450 = buf447; del buf447  # reuse
    cpp_fused_convolution_backward_threshold_backward_84(c_void_p(buf450.data_ptr()), c_void_p(relu_23.data_ptr()))
    del relu_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf451 = aten.convolution_backward(buf450, mean_5, primals_165, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_5
    del primals_165
    buf452 = buf451[0]
    buf458 = buf440; del buf440  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_85(c_void_p(relu_22.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_39.data_ptr()), c_void_p(buf458.data_ptr()))
    del primals_39
    # Source Nodes: [sigmoid_5], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf459 = aten.convolution_backward(buf458, relu_21, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del primals_164
    buf460 = buf459[0]
    buf465 = buf458; del buf458  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_86(c_void_p(relu_21.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_37.data_ptr()), c_void_p(buf465.data_ptr()))
    del primals_37
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf466 = aten.convolution_backward(buf465, relu_20, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf465
    del primals_163
    buf467 = buf466[0]
    buf437 = reinterpret_tensor(buf450, (448, ), (1, ), 0); del buf450  # reuse
    buf438 = empty((448, ), device='cpu', dtype=torch.float32)
    buf469 = empty((448, ), device='cpu', dtype=torch.float32)
    buf470 = empty((448, ), device='cpu', dtype=torch.float32)
    buf439 = buf438; del buf438  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_87(c_void_p(buf439.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_306.data_ptr()), c_void_p(buf437.data_ptr()), c_void_p(buf469.data_ptr()), c_void_p(buf470.data_ptr()))
    del convolution_27
    del convolution_32
    del primals_299
    del primals_305
    del primals_306
    buf443 = buf441[1]
    del buf441
    buf448 = buf446[1]
    buf449 = buf446[2]
    del buf446
    buf453 = buf451[1]
    buf454 = buf451[2]
    del buf451
    buf455 = empty((448, ), device='cpu', dtype=torch.float32)
    buf456 = empty((448, ), device='cpu', dtype=torch.float32)
    buf457 = buf456; del buf456  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_88(c_void_p(buf457.data_ptr()), c_void_p(relu_22.data_ptr()), c_void_p(buf442.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(buf452.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_303.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(buf455.data_ptr()))
    del buf442
    del convolution_29
    del convolution_31
    del primals_303
    del primals_304
    del relu_22
    buf461 = buf459[1]
    del buf459
    buf462 = empty((448, ), device='cpu', dtype=torch.float32)
    buf463 = empty((448, ), device='cpu', dtype=torch.float32)
    buf464 = buf463; del buf463  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_89(c_void_p(buf464.data_ptr()), c_void_p(relu_21.data_ptr()), c_void_p(buf460.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(buf462.data_ptr()))
    del convolution_28
    del primals_301
    del primals_302
    del relu_21
    buf468 = buf466[1]
    del buf466
    buf471 = buf470; del buf470  # reuse
    buf472 = buf460; del buf460  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_90(c_void_p(buf471.data_ptr()), c_void_p(primals_300.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf436.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(primals_35.data_ptr()), c_void_p(buf472.data_ptr()))
    del primals_300
    del primals_35
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf473 = aten.convolution_backward(buf472, mul_55, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf472
    del mul_55
    del primals_162
    buf474 = buf473[0]
    buf475 = buf473[1]
    del buf473
    buf476 = reinterpret_tensor(buf452, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf452  # reuse
    buf477 = reinterpret_tensor(buf476, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf476  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_91(c_void_p(buf477.data_ptr()), c_void_p(buf474.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_26.data_ptr()))
    # Source Nodes: [sigmoid_4], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf478 = aten.convolution_backward(buf477, relu_19, primals_160, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf477
    del primals_160
    buf479 = buf478[0]
    buf480 = buf478[1]
    buf481 = buf478[2]
    del buf478
    buf482 = buf479; del buf479  # reuse
    cpp_fused_convolution_backward_threshold_backward_92(c_void_p(buf482.data_ptr()), c_void_p(relu_19.data_ptr()))
    del relu_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf483 = aten.convolution_backward(buf482, mean_4, primals_158, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_4
    del primals_158
    buf484 = buf483[0]
    buf485 = buf483[1]
    buf486 = buf483[2]
    del buf483
    buf487 = reinterpret_tensor(buf482, (448, ), (1, ), 0); del buf482  # reuse
    buf488 = empty((448, ), device='cpu', dtype=torch.float32)
    buf489 = buf488; del buf488  # reuse
    buf490 = buf474; del buf474  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_93(c_void_p(buf489.data_ptr()), c_void_p(buf490.data_ptr()), c_void_p(relu_18.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(buf484.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_297.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(primals_33.data_ptr()), c_void_p(buf487.data_ptr()))
    del convolution_24
    del convolution_26
    del primals_297
    del primals_298
    del primals_33
    del relu_18
    # Source Nodes: [sigmoid_4], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf491 = aten.convolution_backward(buf490, relu_17, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf490
    del primals_157
    buf492 = buf491[0]
    buf493 = buf491[1]
    del buf491
    buf494 = empty((448, ), device='cpu', dtype=torch.float32)
    buf495 = empty((448, ), device='cpu', dtype=torch.float32)
    buf496 = buf495; del buf495  # reuse
    buf497 = buf492; del buf492  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_94(c_void_p(buf496.data_ptr()), c_void_p(buf497.data_ptr()), c_void_p(relu_17.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_31.data_ptr()), c_void_p(buf494.data_ptr()))
    del convolution_23
    del primals_295
    del primals_296
    del primals_31
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf498 = aten.convolution_backward(buf497, relu_16, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_156
    buf499 = buf498[0]
    buf500 = buf498[1]
    del buf498
    buf501 = buf436; del buf436  # reuse
    buf505 = buf497; del buf497  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_95(c_void_p(buf501.data_ptr()), c_void_p(relu_16.data_ptr()), c_void_p(relu_20.data_ptr()), c_void_p(buf467.data_ptr()), c_void_p(buf499.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf505.data_ptr()))
    del buf467
    del buf499
    del primals_29
    del relu_16
    del relu_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf506 = aten.convolution_backward(buf505, mul_45, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_45
    del primals_155
    buf507 = buf506[0]
    buf509 = reinterpret_tensor(buf484, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf484  # reuse
    buf510 = reinterpret_tensor(buf509, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf509  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_96(c_void_p(buf510.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(convolution_21.data_ptr()))
    # Source Nodes: [sigmoid_3], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf511 = aten.convolution_backward(buf510, relu_15, primals_153, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf510
    del primals_153
    buf512 = buf511[0]
    buf515 = buf512; del buf512  # reuse
    cpp_fused_convolution_backward_threshold_backward_97(c_void_p(buf515.data_ptr()), c_void_p(relu_15.data_ptr()))
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf516 = aten.convolution_backward(buf515, mean_3, primals_151, [112], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_151
    buf517 = buf516[0]
    buf523 = buf505; del buf505  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_98(c_void_p(relu_14.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_27.data_ptr()), c_void_p(buf523.data_ptr()))
    del primals_27
    # Source Nodes: [sigmoid_3], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf524 = aten.convolution_backward(buf523, relu_13, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del primals_150
    buf525 = buf524[0]
    buf530 = buf523; del buf523  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_99(c_void_p(relu_13.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(primals_25.data_ptr()), c_void_p(buf530.data_ptr()))
    del primals_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf531 = aten.convolution_backward(buf530, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf530
    del primals_149
    buf532 = buf531[0]
    buf502 = reinterpret_tensor(buf515, (448, ), (1, ), 0); del buf515  # reuse
    buf503 = empty((448, ), device='cpu', dtype=torch.float32)
    buf534 = empty((448, ), device='cpu', dtype=torch.float32)
    buf535 = empty((448, ), device='cpu', dtype=torch.float32)
    buf541 = empty((448, ), device='cpu', dtype=torch.float32)
    buf504 = buf503; del buf503  # reuse
    cpp_fused_add_native_batch_norm_backward_threshold_backward_100(c_void_p(buf504.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(convolution_22.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(convolution_17.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(convolution_16.data_ptr()), c_void_p(primals_285.data_ptr()), c_void_p(primals_294.data_ptr()), c_void_p(buf502.data_ptr()), c_void_p(buf534.data_ptr()), c_void_p(buf535.data_ptr()), c_void_p(buf541.data_ptr()))
    del convolution_16
    del convolution_17
    del convolution_22
    del primals_285
    del primals_287
    del primals_293
    del primals_294
    buf508 = buf506[1]
    del buf506
    buf513 = buf511[1]
    buf514 = buf511[2]
    del buf511
    buf518 = buf516[1]
    buf519 = buf516[2]
    del buf516
    buf520 = empty((448, ), device='cpu', dtype=torch.float32)
    buf521 = empty((448, ), device='cpu', dtype=torch.float32)
    buf522 = buf521; del buf521  # reuse
    cpp_fused_add_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_101(c_void_p(buf522.data_ptr()), c_void_p(relu_14.data_ptr()), c_void_p(buf507.data_ptr()), c_void_p(convolution_21.data_ptr()), c_void_p(buf517.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_291.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(buf520.data_ptr()))
    del convolution_19
    del convolution_21
    del primals_291
    del primals_292
    del relu_14
    buf526 = buf524[1]
    del buf524
    buf527 = empty((448, ), device='cpu', dtype=torch.float32)
    buf528 = empty((448, ), device='cpu', dtype=torch.float32)
    buf529 = buf528; del buf528  # reuse
    cpp_fused_native_batch_norm_backward_threshold_backward_102(c_void_p(buf529.data_ptr()), c_void_p(relu_13.data_ptr()), c_void_p(buf525.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf527.data_ptr()))
    del convolution_18
    del primals_289
    del primals_290
    del relu_13
    buf533 = buf531[1]
    del buf531
    buf536 = buf535; del buf535  # reuse
    buf537 = buf525; del buf525  # reuse
    buf543 = buf507; del buf507  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_103(c_void_p(buf536.data_ptr()), c_void_p(primals_288.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(buf501.data_ptr()), c_void_p(buf532.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_21.data_ptr()), c_void_p(buf537.data_ptr()), c_void_p(buf543.data_ptr()))
    del buf501
    del buf532
    del primals_21
    del primals_23
    del primals_288
    del relu_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf538 = aten.convolution_backward(buf537, relu_8, primals_148, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf537
    del primals_148
    buf539 = buf538[0]
    buf540 = buf538[1]
    del buf538
    buf542 = buf541; del buf541  # reuse
    cpp_fused_native_batch_norm_backward_104(c_void_p(buf542.data_ptr()), c_void_p(primals_286.data_ptr()))
    del primals_286
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf544 = aten.convolution_backward(buf543, mul_32, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf543
    del mul_32
    del primals_147
    buf545 = buf544[0]
    buf546 = buf544[1]
    del buf544
    buf547 = reinterpret_tensor(buf517, (4, 448, 1, 1), (448, 1, 1792, 1792), 0); del buf517  # reuse
    buf548 = reinterpret_tensor(buf547, (4, 448, 1, 1), (448, 1, 1, 1), 0); del buf547  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_105(c_void_p(buf548.data_ptr()), c_void_p(buf545.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_15.data_ptr()))
    # Source Nodes: [sigmoid_2], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf549 = aten.convolution_backward(buf548, relu_11, primals_145, [448], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf548
    del primals_145
    buf550 = buf549[0]
    buf551 = buf549[1]
    buf552 = buf549[2]
    del buf549
    buf553 = buf550; del buf550  # reuse
    cpp_fused_convolution_backward_threshold_backward_106(c_void_p(buf553.data_ptr()), c_void_p(relu_11.data_ptr()))
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf554 = aten.convolution_backward(buf553, mean_2, primals_143, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_2
    del primals_143
    buf555 = buf554[0]
    buf556 = buf554[1]
    buf557 = buf554[2]
    del buf554
    buf558 = empty((448, ), device='cpu', dtype=torch.float32)
    buf559 = empty((448, ), device='cpu', dtype=torch.float32)
    buf560 = buf559; del buf559  # reuse
    buf561 = buf545; del buf545  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_107(c_void_p(buf560.data_ptr()), c_void_p(buf561.data_ptr()), c_void_p(relu_10.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(buf555.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_19.data_ptr()), c_void_p(buf558.data_ptr()))
    del buf555
    del convolution_13
    del convolution_15
    del primals_19
    del primals_283
    del primals_284
    del relu_10
    # Source Nodes: [sigmoid_2], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf562 = aten.convolution_backward(buf561, relu_9, primals_142, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 4, [True, True, False])
    del buf561
    del primals_142
    buf563 = buf562[0]
    buf564 = buf562[1]
    del buf562
    buf565 = empty((448, ), device='cpu', dtype=torch.float32)
    buf566 = empty((448, ), device='cpu', dtype=torch.float32)
    buf567 = buf566; del buf566  # reuse
    buf568 = buf563; del buf563  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_108(c_void_p(buf567.data_ptr()), c_void_p(buf568.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(convolution_12.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_282.data_ptr()), c_void_p(primals_17.data_ptr()), c_void_p(buf565.data_ptr()))
    del convolution_12
    del primals_17
    del primals_281
    del primals_282
    del relu_9
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf569 = aten.convolution_backward(buf568, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf568
    del primals_141
    buf570 = buf569[0]
    buf571 = buf569[1]
    del buf569
    buf572 = reinterpret_tensor(buf553, (224, ), (1, ), 0); del buf553  # reuse
    buf573 = empty((224, ), device='cpu', dtype=torch.float32)
    buf574 = buf573; del buf573  # reuse
    buf575 = reinterpret_tensor(buf400, (4, 224, 56, 56), (702464, 1, 12544, 224), 0); del buf400  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_109(c_void_p(buf574.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf539.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(convolution_11.data_ptr()), c_void_p(primals_279.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(primals_15.data_ptr()), c_void_p(buf572.data_ptr()), c_void_p(buf575.data_ptr()))
    del convolution_11
    del primals_15
    del primals_279
    del primals_280
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf576 = aten.convolution_backward(buf575, mul_22, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf575
    del mul_22
    del primals_140
    buf577 = buf576[0]
    buf578 = buf576[1]
    del buf576
    buf579 = empty_strided((4, 224, 1, 1), (224, 1, 896, 896), device='cpu', dtype=torch.float32)
    buf580 = reinterpret_tensor(buf579, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf579  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_110(c_void_p(buf580.data_ptr()), c_void_p(buf577.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_10.data_ptr()))
    # Source Nodes: [sigmoid_1], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf581 = aten.convolution_backward(buf580, relu_7, primals_138, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf580
    del primals_138
    buf582 = buf581[0]
    buf583 = buf581[1]
    buf584 = buf581[2]
    del buf581
    buf585 = buf582; del buf582  # reuse
    cpp_fused_convolution_backward_threshold_backward_111(c_void_p(buf585.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf586 = aten.convolution_backward(buf585, mean_1, primals_136, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_1
    del primals_136
    buf587 = buf586[0]
    buf588 = buf586[1]
    buf589 = buf586[2]
    del buf586
    buf590 = reinterpret_tensor(buf585, (224, ), (1, ), 0); del buf585  # reuse
    buf591 = empty((224, ), device='cpu', dtype=torch.float32)
    buf592 = buf591; del buf591  # reuse
    buf593 = buf577; del buf577  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_112(c_void_p(buf592.data_ptr()), c_void_p(buf593.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(buf587.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_13.data_ptr()), c_void_p(buf590.data_ptr()))
    del convolution_10
    del convolution_8
    del primals_13
    del primals_277
    del primals_278
    del relu_6
    # Source Nodes: [sigmoid_1], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf594 = aten.convolution_backward(buf593, relu_5, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del primals_135
    buf595 = buf594[0]
    buf596 = buf594[1]
    del buf594
    buf597 = empty((224, ), device='cpu', dtype=torch.float32)
    buf598 = empty((224, ), device='cpu', dtype=torch.float32)
    buf599 = buf598; del buf598  # reuse
    buf600 = buf595; del buf595  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_113(c_void_p(buf599.data_ptr()), c_void_p(buf600.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_276.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf597.data_ptr()))
    del convolution_7
    del primals_11
    del primals_275
    del primals_276
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf601 = aten.convolution_backward(buf600, relu_4, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_134
    buf602 = buf601[0]
    buf603 = buf601[1]
    del buf601
    buf604 = buf539; del buf539  # reuse
    buf605 = empty((224, ), device='cpu', dtype=torch.float32)
    buf606 = empty((224, ), device='cpu', dtype=torch.float32)
    buf612 = empty((224, ), device='cpu', dtype=torch.float32)
    buf607 = buf606; del buf606  # reuse
    buf608 = buf600; del buf600  # reuse
    buf614 = buf593; del buf593  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_114(c_void_p(buf604.data_ptr()), c_void_p(buf607.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf570.data_ptr()), c_void_p(buf602.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_273.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_9.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(primals_7.data_ptr()), c_void_p(buf605.data_ptr()), c_void_p(buf612.data_ptr()), c_void_p(buf608.data_ptr()), c_void_p(buf614.data_ptr()))
    del buf570
    del buf602
    del buf604
    del convolution_5
    del convolution_6
    del primals_271
    del primals_273
    del primals_274
    del primals_7
    del primals_9
    del relu_4
    del relu_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf609 = aten.convolution_backward(buf608, relu, primals_133, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf608
    del primals_133
    buf610 = buf609[0]
    buf611 = buf609[1]
    del buf609
    buf613 = buf612; del buf612  # reuse
    cpp_fused_native_batch_norm_backward_115(c_void_p(buf613.data_ptr()), c_void_p(primals_272.data_ptr()))
    del primals_272
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf615 = aten.convolution_backward(buf614, mul_9, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf614
    del mul_9
    del primals_132
    buf616 = buf615[0]
    buf617 = buf615[1]
    del buf615
    buf618 = reinterpret_tensor(buf587, (4, 224, 1, 1), (224, 1, 896, 896), 0); del buf587  # reuse
    buf619 = reinterpret_tensor(buf618, (4, 224, 1, 1), (224, 1, 1, 1), 0); del buf618  # reuse
    cpp_fused_convolution_backward_mul_sigmoid_sigmoid_backward_sum_116(c_void_p(buf619.data_ptr()), c_void_p(buf616.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()))
    # Source Nodes: [sigmoid], Original ATen: [aten.convolution_backward, aten.sigmoid, aten.sigmoid_backward]
    buf620 = aten.convolution_backward(buf619, relu_3, primals_130, [224], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf619
    del primals_130
    buf621 = buf620[0]
    buf622 = buf620[1]
    buf623 = buf620[2]
    del buf620
    buf624 = buf621; del buf621  # reuse
    cpp_fused_convolution_backward_threshold_backward_117(c_void_p(buf624.data_ptr()), c_void_p(relu_3.data_ptr()))
    del relu_3
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.threshold_backward]
    buf625 = aten.convolution_backward(buf624, mean, primals_128, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean
    del primals_128
    buf626 = buf625[0]
    buf627 = buf625[1]
    buf628 = buf625[2]
    del buf625
    buf629 = empty((224, ), device='cpu', dtype=torch.float32)
    buf630 = empty((224, ), device='cpu', dtype=torch.float32)
    buf631 = buf630; del buf630  # reuse
    buf632 = buf616; del buf616  # reuse
    cpp_fused_add_convolution_backward_div_mul_native_batch_norm_backward_sigmoid_threshold_backward_118(c_void_p(buf631.data_ptr()), c_void_p(buf632.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(buf626.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_270.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf629.data_ptr()))
    del buf626
    del convolution_2
    del convolution_4
    del primals_269
    del primals_270
    del primals_5
    del relu_2
    # Source Nodes: [sigmoid], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.mul, aten.native_batch_norm_backward, aten.sigmoid, aten.threshold_backward]
    buf633 = aten.convolution_backward(buf632, relu_1, primals_127, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 2, [True, True, False])
    del buf632
    del primals_127
    buf634 = buf633[0]
    buf635 = buf633[1]
    del buf633
    buf636 = empty((224, ), device='cpu', dtype=torch.float32)
    buf637 = empty((224, ), device='cpu', dtype=torch.float32)
    buf638 = buf637; del buf637  # reuse
    buf639 = buf634; del buf634  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_threshold_backward_119(c_void_p(buf638.data_ptr()), c_void_p(buf639.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_267.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_3.data_ptr()), c_void_p(buf636.data_ptr()))
    del convolution_1
    del primals_267
    del primals_268
    del primals_3
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf640 = aten.convolution_backward(buf639, relu, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf639
    del primals_126
    buf641 = buf640[0]
    buf642 = buf640[1]
    del buf640
    buf643 = reinterpret_tensor(buf624, (32, ), (1, ), 0); del buf624  # reuse
    buf644 = empty((32, ), device='cpu', dtype=torch.float32)
    buf645 = buf644; del buf644  # reuse
    buf646 = buf610; del buf610  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_threshold_backward_120(c_void_p(buf645.data_ptr()), c_void_p(buf646.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf641.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_1.data_ptr()), c_void_p(buf643.data_ptr()))
    del buf641
    del convolution
    del primals_1
    del primals_265
    del primals_266
    del relu
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf647 = aten.convolution_backward(buf646, primals_389, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf646
    del primals_125
    del primals_389
    buf648 = buf647[1]
    return (buf645, buf643, buf638, buf636, buf631, buf629, buf613, buf605, buf607, buf605, buf599, buf597, buf592, buf590, buf574, buf572, buf567, buf565, buf560, buf558, buf542, buf534, buf536, buf534, buf529, buf527, buf522, buf520, buf504, buf502, buf496, buf494, buf489, buf487, buf471, buf469, buf464, buf462, buf457, buf455, buf439, buf437, buf431, buf429, buf424, buf422, buf406, buf404, buf399, buf397, buf392, buf390, buf374, buf366, buf368, buf366, buf361, buf359, buf354, buf352, buf336, buf334, buf328, buf326, buf321, buf319, buf303, buf301, buf296, buf294, buf289, buf287, buf271, buf269, buf263, buf261, buf256, buf254, buf238, buf236, buf231, buf229, buf224, buf222, buf206, buf204, buf198, buf196, buf191, buf189, buf173, buf171, buf166, buf164, buf159, buf157, buf141, buf139, buf133, buf131, buf126, buf124, buf108, buf106, buf101, buf99, buf94, buf92, buf76, buf74, buf68, buf66, buf61, buf59, buf43, buf41, buf36, buf34, buf29, buf27, buf11, buf3, buf5, buf3, buf648, buf642, buf635, buf627, buf628, buf622, buf623, buf617, buf611, buf603, buf596, buf588, buf589, buf583, buf584, buf578, buf571, buf564, buf556, buf557, buf551, buf552, buf546, buf540, buf533, buf526, buf518, buf519, buf513, buf514, buf508, buf500, buf493, buf485, buf486, buf480, buf481, buf475, buf468, buf461, buf453, buf454, buf448, buf449, buf443, buf435, buf428, buf420, buf421, buf415, buf416, buf410, buf403, buf396, buf388, buf389, buf383, buf384, buf378, buf372, buf365, buf358, buf350, buf351, buf345, buf346, buf340, buf332, buf325, buf317, buf318, buf312, buf313, buf307, buf300, buf293, buf285, buf286, buf280, buf281, buf275, buf267, buf260, buf252, buf253, buf247, buf248, buf242, buf235, buf228, buf220, buf221, buf215, buf216, buf210, buf202, buf195, buf187, buf188, buf182, buf183, buf177, buf170, buf163, buf155, buf156, buf150, buf151, buf145, buf137, buf130, buf122, buf123, buf117, buf118, buf112, buf105, buf98, buf90, buf91, buf85, buf86, buf80, buf72, buf65, buf57, buf58, buf52, buf53, buf47, buf40, buf33, buf25, buf26, buf20, buf21, buf15, buf9, reinterpret_tensor(buf1, (1000, 2240), (2240, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_3 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_9 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_15 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_21 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_27 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_33 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_35 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_37 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_43 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_53 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_59 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_61 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_63 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_69 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_75 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_81 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_87 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_93 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_99 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_105 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_115 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_121 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_123 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_125 = rand_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((224, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_128 = rand_strided((8, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((224, 8, 1, 1), (8, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((224, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_134 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((224, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_136 = rand_strided((56, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_138 = rand_strided((224, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((224, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_141 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((56, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((448, 56, 1, 1), (56, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_147 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((448, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_149 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_151 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_157 = rand_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_160 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_162 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_164 = rand_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_167 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_170 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_171 = rand_strided((448, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_172 = rand_strided((112, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_174 = rand_strided((448, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((448, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_177 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((112, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((896, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_183 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((896, 448, 1, 1), (448, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_186 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_189 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_192 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_198 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_201 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_207 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_210 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_213 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_219 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_222 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_228 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_231 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_234 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_240 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_243 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_249 = rand_strided((896, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((224, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_252 = rand_strided((896, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((896, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_255 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((2240, 112, 3, 3), (1008, 1, 336, 112), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((224, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((2240, 224, 1, 1), (224, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_261 = rand_strided((2240, 2240, 1, 1), (2240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((2240, 896, 1, 1), (896, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((32, ), (1, ), device='cpu', dtype=torch.float32)
    primals_267 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_270 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_273 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_276 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_279 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((224, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_282 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_285 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_288 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_291 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_294 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_297 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_300 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_303 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_306 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_309 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_312 = rand_strided((448, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_314 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_315 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_316 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_317 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_318 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_319 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_320 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_321 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_322 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_323 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_324 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_325 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_326 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_327 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_328 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_329 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_330 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_331 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_332 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_333 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_334 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_335 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_336 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_337 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_338 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_339 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_340 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_341 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_342 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_343 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_344 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_345 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_346 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_347 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_348 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_349 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_350 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_351 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_352 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_353 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_354 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_355 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_356 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_357 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_358 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_359 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_360 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_361 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_362 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_363 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_364 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_365 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_366 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_367 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_368 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_369 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_370 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_371 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_372 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_373 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_374 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_375 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_376 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_377 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_378 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_379 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_380 = rand_strided((896, ), (1, ), device='cpu', dtype=torch.float32)
    primals_381 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_382 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_383 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_384 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_385 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_386 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_387 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_388 = rand_strided((2240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_389 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 224, 112, 112), (2809856, 1, 25088, 224), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 224, 112, 112), (2809856, 1, 25088, 224), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 8, 1, 1), (8, 1, 8, 8), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    mul_9 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    mul_22 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_11 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 224, 56, 56), (702464, 1, 12544, 224), device='cpu', dtype=torch.float32)
    convolution_12 = rand_strided((4, 448, 56, 56), (1404928, 1, 25088, 448), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 448, 56, 56), (1404928, 1, 25088, 448), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 56, 1, 1), (56, 1, 56, 56), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    mul_32 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_16 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_17 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    convolution_21 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    mul_45 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_22 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    relu_19 = rand_strided((4, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    mul_55 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_20 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_21 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_22 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    relu_23 = rand_strided((4, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    mul_65 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_24 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_25 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_26 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    relu_27 = rand_strided((4, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 448, 1, 1), (448, 1, 448, 448), device='cpu', dtype=torch.float32)
    mul_75 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    relu_28 = rand_strided((4, 448, 28, 28), (351232, 1, 12544, 448), device='cpu', dtype=torch.float32)
    convolution_38 = rand_strided((4, 896, 28, 28), (702464, 1, 25088, 896), device='cpu', dtype=torch.float32)
    relu_29 = rand_strided((4, 896, 28, 28), (702464, 1, 25088, 896), device='cpu', dtype=torch.float32)
    convolution_39 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_30 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_31 = rand_strided((4, 112, 1, 1), (112, 1, 112, 112), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_85 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_43 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_32 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_44 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_33 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_34 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_8 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_35 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_98 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_48 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_36 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_49 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_37 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_38 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_9 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_39 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_108 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_53 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_40 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_54 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_41 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_42 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_10 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_43 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_118 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_58 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_44 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_59 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_45 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_46 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_11 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_47 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_62 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_128 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_63 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_48 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_64 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_49 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_65 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_50 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_12 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_51 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_67 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_138 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_68 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_52 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_69 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_53 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_70 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_54 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_13 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_55 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_72 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_148 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_73 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_56 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_74 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_57 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_75 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_58 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_14 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_59 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_77 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_158 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_78 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_60 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_79 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_61 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_80 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_62 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_15 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_63 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_82 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_168 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_83 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_64 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_84 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_65 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_85 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_66 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_16 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_67 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_87 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_178 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_88 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_68 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_89 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_69 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_90 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_70 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    mean_17 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    relu_71 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_92 = rand_strided((4, 896, 1, 1), (896, 1, 896, 896), device='cpu', dtype=torch.float32)
    mul_188 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_93 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    relu_72 = rand_strided((4, 896, 14, 14), (175616, 1, 12544, 896), device='cpu', dtype=torch.float32)
    convolution_94 = rand_strided((4, 2240, 14, 14), (439040, 1, 31360, 2240), device='cpu', dtype=torch.float32)
    relu_73 = rand_strided((4, 2240, 14, 14), (439040, 1, 31360, 2240), device='cpu', dtype=torch.float32)
    convolution_95 = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    relu_74 = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    mean_18 = rand_strided((4, 2240, 1, 1), (2240, 1, 2240, 2240), device='cpu', dtype=torch.float32)
    relu_75 = rand_strided((4, 224, 1, 1), (224, 1, 224, 224), device='cpu', dtype=torch.float32)
    convolution_97 = rand_strided((4, 2240, 1, 1), (2240, 1, 2240, 2240), device='cpu', dtype=torch.float32)
    mul_198 = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    convolution_98 = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    convolution_99 = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 2240), (2240, 1), device='cpu', dtype=torch.float32)
    permute_1 = rand_strided((1000, 2240), (2240, 1), device='cpu', dtype=torch.float32)
    le = rand_strided((4, 2240, 7, 7), (109760, 1, 15680, 2240), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_126, primals_127, primals_128, primals_130, primals_132, primals_133, primals_134, primals_135, primals_136, primals_138, primals_140, primals_141, primals_142, primals_143, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_155, primals_156, primals_157, primals_158, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_169, primals_170, primals_171, primals_172, primals_174, primals_176, primals_177, primals_178, primals_179, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_191, primals_192, primals_193, primals_194, primals_196, primals_198, primals_199, primals_200, primals_201, primals_203, primals_205, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_226, primals_227, primals_228, primals_229, primals_231, primals_233, primals_234, primals_235, primals_236, primals_238, primals_240, primals_241, primals_242, primals_243, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_254, primals_255, primals_256, primals_257, primals_259, primals_261, primals_262, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, convolution, relu, convolution_1, relu_1, convolution_2, relu_2, mean, relu_3, convolution_4, mul_9, convolution_5, convolution_6, relu_4, convolution_7, relu_5, convolution_8, relu_6, mean_1, relu_7, convolution_10, mul_22, convolution_11, relu_8, convolution_12, relu_9, convolution_13, relu_10, mean_2, relu_11, convolution_15, mul_32, convolution_16, convolution_17, relu_12, convolution_18, relu_13, convolution_19, relu_14, mean_3, relu_15, convolution_21, mul_45, convolution_22, relu_16, convolution_23, relu_17, convolution_24, relu_18, mean_4, relu_19, convolution_26, mul_55, convolution_27, relu_20, convolution_28, relu_21, convolution_29, relu_22, mean_5, relu_23, convolution_31, mul_65, convolution_32, relu_24, convolution_33, relu_25, convolution_34, relu_26, mean_6, relu_27, convolution_36, mul_75, convolution_37, relu_28, convolution_38, relu_29, convolution_39, relu_30, mean_7, relu_31, convolution_41, mul_85, convolution_42, convolution_43, relu_32, convolution_44, relu_33, convolution_45, relu_34, mean_8, relu_35, convolution_47, mul_98, convolution_48, relu_36, convolution_49, relu_37, convolution_50, relu_38, mean_9, relu_39, convolution_52, mul_108, convolution_53, relu_40, convolution_54, relu_41, convolution_55, relu_42, mean_10, relu_43, convolution_57, mul_118, convolution_58, relu_44, convolution_59, relu_45, convolution_60, relu_46, mean_11, relu_47, convolution_62, mul_128, convolution_63, relu_48, convolution_64, relu_49, convolution_65, relu_50, mean_12, relu_51, convolution_67, mul_138, convolution_68, relu_52, convolution_69, relu_53, convolution_70, relu_54, mean_13, relu_55, convolution_72, mul_148, convolution_73, relu_56, convolution_74, relu_57, convolution_75, relu_58, mean_14, relu_59, convolution_77, mul_158, convolution_78, relu_60, convolution_79, relu_61, convolution_80, relu_62, mean_15, relu_63, convolution_82, mul_168, convolution_83, relu_64, convolution_84, relu_65, convolution_85, relu_66, mean_16, relu_67, convolution_87, mul_178, convolution_88, relu_68, convolution_89, relu_69, convolution_90, relu_70, mean_17, relu_71, convolution_92, mul_188, convolution_93, relu_72, convolution_94, relu_73, convolution_95, relu_74, mean_18, relu_75, convolution_97, mul_198, convolution_98, convolution_99, clone, permute_1, le, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_regnet', benchmark_compiled_module)
