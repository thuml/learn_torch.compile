
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


cpp_fused_hardswish_backward_sum_view_0 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5120L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
            auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = static_cast<float>(-3.0);
            auto tmp2 = at::vec::Vectorized<float>(tmp1);
            auto tmp3 = to_float_mask(tmp0 < tmp2);
            auto tmp4 = static_cast<float>(3.0);
            auto tmp5 = at::vec::Vectorized<float>(tmp4);
            auto tmp6 = to_float_mask(tmp0 <= tmp5);
            auto tmp8 = tmp0 / tmp5;
            auto tmp9 = static_cast<float>(0.5);
            auto tmp10 = at::vec::Vectorized<float>(tmp9);
            auto tmp11 = tmp8 + tmp10;
            auto tmp12 = tmp7 * tmp11;
            auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
            auto tmp14 = static_cast<float>(0.0);
            auto tmp15 = at::vec::Vectorized<float>(tmp14);
            auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
            tmp16.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_sum_view_1 = async_compile.cpp('''
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
                       float* out_ptr1,
                       float* out_ptr3)
{
    auto out_ptr2 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1280L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(1280L + x0));
            auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(2560L + x0));
            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(3840L + x0));
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp21 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp8 = static_cast<float>(49.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp0 / tmp5;
                            auto tmp12 = static_cast<float>(0.5);
                            auto tmp13 = at::vec::Vectorized<float>(tmp12);
                            auto tmp14 = tmp11 + tmp13;
                            auto tmp15 = tmp10 * tmp14;
                            auto tmp16 = decltype(tmp15)::blendv(tmp10, tmp15, tmp6);
                            auto tmp17 = static_cast<float>(0.0);
                            auto tmp18 = at::vec::Vectorized<float>(tmp17);
                            auto tmp19 = decltype(tmp18)::blendv(tmp16, tmp18, tmp3);
                            auto tmp22 = tmp20 - tmp21;
                            auto tmp23 = tmp19 * tmp22;
                            tmp_acc0_vec = tmp_acc0_vec + tmp19;
                            tmp_acc1_vec = tmp_acc1_vec + tmp23;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = static_cast<float>(49.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp0 / tmp5;
                        auto tmp12 = static_cast<float>(0.5);
                        auto tmp13 = at::vec::Vectorized<float>(tmp12);
                        auto tmp14 = tmp11 + tmp13;
                        auto tmp15 = tmp10 * tmp14;
                        auto tmp16 = decltype(tmp15)::blendv(tmp10, tmp15, tmp6);
                        auto tmp17 = static_cast<float>(0.0);
                        auto tmp18 = at::vec::Vectorized<float>(tmp17);
                        auto tmp19 = decltype(tmp18)::blendv(tmp16, tmp18, tmp3);
                        auto tmp21 = static_cast<float>(0.001);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = tmp20 + tmp22;
                        auto tmp24 = tmp23.rsqrt();
                        auto tmp26 = tmp24 * tmp25;
                        auto tmp27 = tmp19 * tmp26;
                        tmp27.store(out_ptr3 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_2 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            {
                #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                float tmp_acc0 = 0;
                at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                float tmp_acc1 = 0;
                at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.001);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr2 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_3 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_4 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_5 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(49.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp15 = tmp0 / tmp5;
                            auto tmp16 = static_cast<float>(0.5);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = tmp14 * tmp18;
                            auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                            auto tmp26 = tmp24 - tmp25;
                            auto tmp27 = tmp23 * tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp23;
                            tmp_acc1_vec = tmp_acc1_vec + tmp27;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp25 = static_cast<float>(0.001);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp31 = tmp23 * tmp30;
                        tmp31.store(in_out_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_6 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(in_out_ptr1 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_7 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(0.001);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(out_ptr0 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_8 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x2) + (47040L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(3840L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_9 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_10 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(49L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(960L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (960L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp25 = static_cast<float>(0.001);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp31 = tmp23 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (960L*x1) + (47040L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(960L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (960L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (960L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_12 = async_compile.cpp('''
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
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (160L*x1)));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp5 = tmp3 - tmp4;
                    auto tmp6 = tmp2 * tmp5;
                    auto tmp8 = tmp2 + tmp7;
                    auto tmp11 = tmp9 - tmp10;
                    auto tmp12 = tmp8 * tmp11;
                    tmp_acc0_vec = tmp_acc0_vec + tmp2;
                    tmp_acc1_vec = tmp_acc1_vec + tmp6;
                    tmp_acc2_vec = tmp_acc2_vec + tmp8;
                    tmp_acc3_vec = tmp_acc3_vec + tmp12;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_13 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (960L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (960L*x2) + (47040L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(49.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp15 = tmp0 / tmp5;
                            auto tmp16 = static_cast<float>(0.5);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = tmp14 * tmp18;
                            auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                            auto tmp26 = tmp24 - tmp25;
                            auto tmp27 = tmp23 * tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp23;
                            tmp_acc1_vec = tmp_acc1_vec + tmp27;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_14 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (960L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(960L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_15 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(160L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(196L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(160L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (160L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (160L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = static_cast<float>(0.001);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (160L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_16 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(49L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (32928L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_17 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_18 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (32928L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(49.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp15 = tmp0 / tmp5;
                            auto tmp16 = static_cast<float>(0.5);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = tmp14 * tmp18;
                            auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                            auto tmp26 = tmp24 - tmp25;
                            auto tmp27 = tmp23 * tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp23;
                            tmp_acc1_vec = tmp_acc1_vec + tmp27;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(49.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp25 = static_cast<float>(0.001);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp31 = tmp23 * tmp30;
                        tmp31.store(in_out_ptr1 + static_cast<long>(x2 + (672L*x1) + (32928L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(in_out_ptr1 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_20 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.001);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_21 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x2) + (131712L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(2688L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_22 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_23 = async_compile.cpp('''
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
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(672L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (672L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(196.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp25 = static_cast<float>(0.001);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp31 = tmp23 * tmp30;
                        tmp31.store(out_ptr0 + static_cast<long>(x2 + (672L*x1) + (131712L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_24 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(672L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (672L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (672L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_25 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
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
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (112L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                    tmp_acc3_vec = tmp_acc3_vec + tmp10;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (672L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (672L*x2) + (131712L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(196.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp15 = tmp0 / tmp5;
                            auto tmp16 = static_cast<float>(0.5);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = tmp14 * tmp18;
                            auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                            auto tmp26 = tmp24 - tmp25;
                            auto tmp27 = tmp23 * tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp23;
                            tmp_acc1_vec = tmp_acc1_vec + tmp27;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_27 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (672L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(672L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_28 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(112L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(112L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (112L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (112L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(0.001);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (112L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_29 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(196L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (480L*x2) + (94080L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_30 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_31 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (480L*x1)));
                            auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (480L*x2) + (94080L*x1)));
                            auto tmp25 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(-3.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 < tmp2);
                            auto tmp4 = static_cast<float>(3.0);
                            auto tmp5 = at::vec::Vectorized<float>(tmp4);
                            auto tmp6 = to_float_mask(tmp0 <= tmp5);
                            auto tmp9 = tmp7 * tmp8;
                            auto tmp11 = static_cast<float>(196.0);
                            auto tmp12 = at::vec::Vectorized<float>(tmp11);
                            auto tmp13 = tmp10 / tmp12;
                            auto tmp14 = tmp9 + tmp13;
                            auto tmp15 = tmp0 / tmp5;
                            auto tmp16 = static_cast<float>(0.5);
                            auto tmp17 = at::vec::Vectorized<float>(tmp16);
                            auto tmp18 = tmp15 + tmp17;
                            auto tmp19 = tmp14 * tmp18;
                            auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                            auto tmp21 = static_cast<float>(0.0);
                            auto tmp22 = at::vec::Vectorized<float>(tmp21);
                            auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                            auto tmp26 = tmp24 - tmp25;
                            auto tmp27 = tmp23 * tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp23;
                            tmp_acc1_vec = tmp_acc1_vec + tmp27;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(196L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(480L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (480L*x0)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp29 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp9 = tmp7 * tmp8;
                        auto tmp11 = static_cast<float>(196.0);
                        auto tmp12 = at::vec::Vectorized<float>(tmp11);
                        auto tmp13 = tmp10 / tmp12;
                        auto tmp14 = tmp9 + tmp13;
                        auto tmp15 = tmp0 / tmp5;
                        auto tmp16 = static_cast<float>(0.5);
                        auto tmp17 = at::vec::Vectorized<float>(tmp16);
                        auto tmp18 = tmp15 + tmp17;
                        auto tmp19 = tmp14 * tmp18;
                        auto tmp20 = decltype(tmp19)::blendv(tmp14, tmp19, tmp6);
                        auto tmp21 = static_cast<float>(0.0);
                        auto tmp22 = at::vec::Vectorized<float>(tmp21);
                        auto tmp23 = decltype(tmp22)::blendv(tmp20, tmp22, tmp3);
                        auto tmp25 = static_cast<float>(0.001);
                        auto tmp26 = at::vec::Vectorized<float>(tmp25);
                        auto tmp27 = tmp24 + tmp26;
                        auto tmp28 = tmp27.rsqrt();
                        auto tmp30 = tmp28 * tmp29;
                        auto tmp31 = tmp23 * tmp30;
                        tmp31.store(in_out_ptr1 + static_cast<long>(x2 + (480L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (480L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(480L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(in_out_ptr1 + static_cast<long>(x1 + (480L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_33 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp2 = static_cast<float>(0.001);
                auto tmp3 = at::vec::Vectorized<float>(tmp2);
                auto tmp4 = tmp1 + tmp3;
                auto tmp5 = tmp4.rsqrt();
                auto tmp7 = tmp5 * tmp6;
                auto tmp8 = tmp0 * tmp7;
                tmp8.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_36 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = static_cast<float>(0.001);
                auto tmp5 = at::vec::Vectorized<float>(tmp4);
                auto tmp6 = tmp3 + tmp5;
                auto tmp7 = tmp6.rsqrt();
                auto tmp9 = tmp7 * tmp8;
                auto tmp10 = tmp2 * tmp9;
                tmp10.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_38 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(184L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (184L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (184L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_39 = async_compile.cpp('''
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = static_cast<float>(0.001);
                auto tmp7 = at::vec::Vectorized<float>(tmp6);
                auto tmp8 = tmp5 + tmp7;
                auto tmp9 = tmp8.rsqrt();
                auto tmp11 = tmp9 * tmp10;
                auto tmp12 = tmp4 * tmp11;
                tmp12.store(out_ptr0 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_40 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_41 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(200L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (200L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(out_ptr0 + static_cast<long>(x1 + (200L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_42 = async_compile.cpp('''
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
                       float* out_ptr0,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       float* out_ptr7)
{
    auto out_ptr1 = in_out_ptr0;
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
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
                float tmp_acc5 = 0;
                at::vec::Vectorized<float> tmp_acc5_vec = at::vec::Vectorized<float>(0);
                float tmp_acc6 = 0;
                at::vec::Vectorized<float> tmp_acc6_vec = at::vec::Vectorized<float>(0);
                float tmp_acc7 = 0;
                at::vec::Vectorized<float> tmp_acc7_vec = at::vec::Vectorized<float>(0);
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr8 + static_cast<long>(x0));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr9 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr10 + static_cast<long>(x0 + (80L*x1)));
                    auto tmp20 = at::vec::Vectorized<float>::loadu(in_ptr11 + static_cast<long>(x0));
                    auto tmp3 = tmp1 - tmp2;
                    auto tmp4 = tmp0 * tmp3;
                    auto tmp6 = tmp0 + tmp5;
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = tmp6 * tmp9;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp15 = tmp13 - tmp14;
                    auto tmp16 = tmp12 * tmp15;
                    auto tmp18 = tmp12 + tmp17;
                    auto tmp21 = tmp19 - tmp20;
                    auto tmp22 = tmp18 * tmp21;
                    tmp_acc0_vec = tmp_acc0_vec + tmp0;
                    tmp_acc1_vec = tmp_acc1_vec + tmp4;
                    tmp_acc2_vec = tmp_acc2_vec + tmp6;
                    tmp_acc3_vec = tmp_acc3_vec + tmp10;
                    tmp_acc4_vec = tmp_acc4_vec + tmp12;
                    tmp_acc5_vec = tmp_acc5_vec + tmp16;
                    tmp_acc6_vec = tmp_acc6_vec + tmp18;
                    tmp_acc7_vec = tmp_acc7_vec + tmp22;
                }
                tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                tmp_acc2_vec.store(out_ptr2 + static_cast<long>(x0));
                tmp_acc3_vec.store(out_ptr3 + static_cast<long>(x0));
                tmp_acc4_vec.store(out_ptr4 + static_cast<long>(x0));
                tmp_acc5_vec.store(out_ptr5 + static_cast<long>(x0));
                tmp_acc6_vec.store(out_ptr6 + static_cast<long>(x0));
                tmp_acc7_vec.store(out_ptr7 + static_cast<long>(x0));
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr12 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardswish_backward_native_batch_norm_backward_43 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_44 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_native_batch_norm_backward_45 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardswish_backward_native_batch_norm_backward_46 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_47 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (184L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(184L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_native_batch_norm_backward_48 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
}
''')


cpp_fused_hardswish_backward_native_batch_norm_backward_49 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_50 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (200L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(200L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_51 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(80L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
            auto tmp3 = at::vec::Vectorized<float>(tmp2);
            auto tmp4 = tmp1 + tmp3;
            auto tmp5 = tmp4.rsqrt();
            auto tmp6 = tmp0 * tmp5;
            tmp6.store(in_out_ptr0 + static_cast<long>(x0));
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(784L); x0+=static_cast<long>(1L))
        {
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(80L); x1+=static_cast<long>(8L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (80L*x0)));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (80L*x0)));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (80L*x0)));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = static_cast<float>(0.001);
                auto tmp9 = at::vec::Vectorized<float>(tmp8);
                auto tmp10 = tmp7 + tmp9;
                auto tmp11 = tmp10.rsqrt();
                auto tmp13 = tmp11 * tmp12;
                auto tmp14 = tmp6 * tmp13;
                tmp14.store(in_out_ptr1 + static_cast<long>(x1 + (80L*x0)));
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_53 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (240L*x1)));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp1 = static_cast<float>(-3.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 < tmp2);
                        auto tmp4 = static_cast<float>(3.0);
                        auto tmp5 = at::vec::Vectorized<float>(tmp4);
                        auto tmp6 = to_float_mask(tmp0 <= tmp5);
                        auto tmp8 = tmp0 / tmp5;
                        auto tmp9 = static_cast<float>(0.5);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = tmp8 + tmp10;
                        auto tmp12 = tmp7 * tmp11;
                        auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                        auto tmp14 = static_cast<float>(0.0);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                        auto tmp19 = tmp17 - tmp18;
                        auto tmp20 = tmp16 * tmp19;
                        tmp_acc0_vec = tmp_acc0_vec + tmp16;
                        tmp_acc1_vec = tmp_acc1_vec + tmp20;
                    }
                    tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x0));
                    tmp_acc1_vec.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(240L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(240L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp22 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp8 = tmp0 / tmp5;
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = at::vec::Vectorized<float>(tmp9);
                    auto tmp11 = tmp8 + tmp10;
                    auto tmp12 = tmp7 * tmp11;
                    auto tmp13 = decltype(tmp12)::blendv(tmp7, tmp12, tmp6);
                    auto tmp14 = static_cast<float>(0.0);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = decltype(tmp15)::blendv(tmp13, tmp15, tmp3);
                    auto tmp18 = static_cast<float>(0.001);
                    auto tmp19 = at::vec::Vectorized<float>(tmp18);
                    auto tmp20 = tmp17 + tmp19;
                    auto tmp21 = tmp20.rsqrt();
                    auto tmp23 = tmp21 * tmp22;
                    auto tmp24 = tmp16 * tmp23;
                    tmp24.store(in_out_ptr1 + static_cast<long>(x1 + (240L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_54 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr2)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.001);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr2 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_55 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_56 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_57 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(784.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp6 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(784.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(0.001);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(in_out_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_58 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_convolution_backward_native_batch_norm_backward_59 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.001);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(out_ptr0 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_60 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x2) + (94080L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(480L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_61 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(128L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_62 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(120L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (120L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(784.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(0.001);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(out_ptr0 + static_cast<long>(x2 + (120L*x1) + (94080L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_63 = async_compile.cpp('''
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(120L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (120L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (120L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_64 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
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
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (40L*x1)));
                        auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp5 = tmp3 - tmp4;
                        auto tmp6 = tmp2 * tmp5;
                        auto tmp8 = tmp2 + tmp7;
                        auto tmp11 = tmp9 - tmp10;
                        auto tmp12 = tmp8 * tmp11;
                        tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        tmp_acc1_vec = tmp_acc1_vec + tmp6;
                        tmp_acc2_vec = tmp_acc2_vec + tmp8;
                        tmp_acc3_vec = tmp_acc3_vec + tmp12;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_65 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (120L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (120L*x2) + (94080L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(784.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp6 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_66 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3136L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (120L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (120L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(120L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_67 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(40L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(40L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (40L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp10 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(0.001);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp5 + tmp7;
                    auto tmp9 = tmp8.rsqrt();
                    auto tmp11 = tmp9 * tmp10;
                    auto tmp12 = tmp4 * tmp11;
                    tmp12.store(in_out_ptr1 + static_cast<long>(x1 + (40L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_68 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const bool* in_ptr2)
{
    auto out_ptr0 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    {
                        #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                        float tmp_acc0 = 0;
                        at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                        for(long x2=static_cast<long>(0L); x2<static_cast<long>(784L); x2+=static_cast<long>(1L))
                        {
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x2) + (56448L*x0)));
                            auto tmp2 = tmp0 * tmp1;
                            tmp_acc0_vec = tmp_acc0_vec + tmp2;
                        }
                        tmp_acc0_vec.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    }
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(288L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = flag_to_float_vec(in_ptr2 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(out_ptr0 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.16666666666666666);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = static_cast<float>(0.0);
                    auto tmp6 = at::vec::Vectorized<float>(tmp5);
                    auto tmp7 = decltype(tmp4)::blendv(tmp6, tmp4, tmp0);
                    tmp7.store(in_out_ptr0 + static_cast<long>(x0));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_threshold_backward_69 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(96L); x0+=static_cast<long>(8L))
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


cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_70 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
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
                            auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x2) + (56448L*x1)));
                            auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x2) + (56448L*x1)));
                            auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
                            auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (72L*x1)));
                            auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (72L*x2) + (56448L*x1)));
                            auto tmp14 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                            auto tmp1 = static_cast<float>(0.0);
                            auto tmp2 = at::vec::Vectorized<float>(tmp1);
                            auto tmp3 = to_float_mask(tmp0 <= tmp2);
                            auto tmp6 = tmp4 * tmp5;
                            auto tmp8 = static_cast<float>(784.0);
                            auto tmp9 = at::vec::Vectorized<float>(tmp8);
                            auto tmp10 = tmp7 / tmp9;
                            auto tmp11 = tmp6 + tmp10;
                            auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                            auto tmp15 = tmp13 - tmp14;
                            auto tmp16 = tmp12 * tmp15;
                            tmp_acc0_vec = tmp_acc0_vec + tmp12;
                            tmp_acc1_vec = tmp_acc1_vec + tmp16;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(784L); x1+=static_cast<long>(1L))
                {
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(72L); x2+=static_cast<long>(8L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x2 + (72L*x0)));
                        auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x2));
                        auto tmp18 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x2));
                        auto tmp1 = static_cast<float>(0.0);
                        auto tmp2 = at::vec::Vectorized<float>(tmp1);
                        auto tmp3 = to_float_mask(tmp0 <= tmp2);
                        auto tmp6 = tmp4 * tmp5;
                        auto tmp8 = static_cast<float>(784.0);
                        auto tmp9 = at::vec::Vectorized<float>(tmp8);
                        auto tmp10 = tmp7 / tmp9;
                        auto tmp11 = tmp6 + tmp10;
                        auto tmp12 = decltype(tmp2)::blendv(tmp11, tmp2, tmp3);
                        auto tmp14 = static_cast<float>(0.001);
                        auto tmp15 = at::vec::Vectorized<float>(tmp14);
                        auto tmp16 = tmp13 + tmp15;
                        auto tmp17 = tmp16.rsqrt();
                        auto tmp19 = tmp17 * tmp18;
                        auto tmp20 = tmp12 * tmp19;
                        tmp20.store(in_out_ptr1 + static_cast<long>(x2 + (72L*x1) + (56448L*x0)));
                    }
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_71 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_72 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.001);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_73 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_74 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(72L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (72L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (72L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_native_batch_norm_backward_75 = async_compile.cpp('''
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
                       float* out_ptr2,
                       float* out_ptr3)
{
    auto out_ptr1 = in_out_ptr0;
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (24L*x1)));
                        auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = tmp0 + tmp5;
                        auto tmp9 = tmp7 - tmp8;
                        auto tmp10 = tmp6 * tmp9;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp6;
                        tmp_acc3_vec = tmp_acc3_vec + tmp10;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_76 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_77 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (72L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (72L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(72L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_native_batch_norm_backward_78 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(24L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(12544L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(24L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (24L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(0.001);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = tmp3 + tmp5;
                    auto tmp7 = tmp6.rsqrt();
                    auto tmp9 = tmp7 * tmp8;
                    auto tmp10 = tmp2 * tmp9;
                    tmp10.store(in_out_ptr1 + static_cast<long>(x1 + (24L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_79 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(12544L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_80 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (64L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (64L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(64L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(64L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(in_out_ptr1 + static_cast<long>(x1 + (64L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_native_batch_norm_backward_81 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(28)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(0.001);
                    auto tmp3 = at::vec::Vectorized<float>(tmp2);
                    auto tmp4 = tmp1 + tmp3;
                    auto tmp5 = tmp4.rsqrt();
                    auto tmp7 = tmp5 * tmp6;
                    auto tmp8 = tmp0 * tmp7;
                    tmp8.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_82 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(0.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 <= tmp2);
                    auto tmp5 = decltype(tmp2)::blendv(tmp4, tmp2, tmp3);
                    auto tmp7 = static_cast<float>(0.001);
                    auto tmp8 = at::vec::Vectorized<float>(tmp7);
                    auto tmp9 = tmp6 + tmp8;
                    auto tmp10 = tmp9.rsqrt();
                    auto tmp12 = tmp10 * tmp11;
                    auto tmp13 = tmp5 * tmp12;
                    tmp13.store(out_ptr0 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


cpp_fused_add_hardswish_backward_native_batch_norm_backward_83 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
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
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp2 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp12 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp23 = at::vec::Vectorized<float>::loadu(in_ptr5 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr6 + static_cast<long>(x0));
                        auto tmp3 = tmp1 - tmp2;
                        auto tmp4 = tmp0 * tmp3;
                        auto tmp6 = static_cast<float>(-3.0);
                        auto tmp7 = at::vec::Vectorized<float>(tmp6);
                        auto tmp8 = to_float_mask(tmp5 < tmp7);
                        auto tmp9 = static_cast<float>(3.0);
                        auto tmp10 = at::vec::Vectorized<float>(tmp9);
                        auto tmp11 = to_float_mask(tmp5 <= tmp10);
                        auto tmp13 = tmp0 + tmp12;
                        auto tmp14 = tmp5 / tmp10;
                        auto tmp15 = static_cast<float>(0.5);
                        auto tmp16 = at::vec::Vectorized<float>(tmp15);
                        auto tmp17 = tmp14 + tmp16;
                        auto tmp18 = tmp13 * tmp17;
                        auto tmp19 = decltype(tmp18)::blendv(tmp13, tmp18, tmp11);
                        auto tmp20 = static_cast<float>(0.0);
                        auto tmp21 = at::vec::Vectorized<float>(tmp20);
                        auto tmp22 = decltype(tmp21)::blendv(tmp19, tmp21, tmp8);
                        auto tmp25 = tmp23 - tmp24;
                        auto tmp26 = tmp22 * tmp25;
                        tmp_acc0_vec = tmp_acc0_vec + tmp0;
                        tmp_acc1_vec = tmp_acc1_vec + tmp4;
                        tmp_acc2_vec = tmp_acc2_vec + tmp22;
                        tmp_acc3_vec = tmp_acc3_vec + tmp26;
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr7 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_84 = async_compile.cpp('''
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    float tmp_acc1 = 0;
                    at::vec::Vectorized<float> tmp_acc1_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(50176L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0 + (16L*x1)));
                        auto tmp6 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0 + (16L*x1)));
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
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr1 + static_cast<long>(x0));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x0));
                    auto tmp2 = static_cast<float>(0.001);
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


cpp_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_85 = async_compile.cpp('''
#include "/tmp/torchinductor_youkaichao/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3)
{
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(16L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
            auto tmp2 = static_cast<float>(0.001);
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
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(50176L); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(16L); x1+=static_cast<long>(8L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_out_ptr1 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp8 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (16L*x0)));
                    auto tmp19 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1));
                    auto tmp24 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = static_cast<float>(-3.0);
                    auto tmp2 = at::vec::Vectorized<float>(tmp1);
                    auto tmp3 = to_float_mask(tmp0 < tmp2);
                    auto tmp4 = static_cast<float>(3.0);
                    auto tmp5 = at::vec::Vectorized<float>(tmp4);
                    auto tmp6 = to_float_mask(tmp0 <= tmp5);
                    auto tmp9 = tmp7 + tmp8;
                    auto tmp10 = tmp0 / tmp5;
                    auto tmp11 = static_cast<float>(0.5);
                    auto tmp12 = at::vec::Vectorized<float>(tmp11);
                    auto tmp13 = tmp10 + tmp12;
                    auto tmp14 = tmp9 * tmp13;
                    auto tmp15 = decltype(tmp14)::blendv(tmp9, tmp14, tmp6);
                    auto tmp16 = static_cast<float>(0.0);
                    auto tmp17 = at::vec::Vectorized<float>(tmp16);
                    auto tmp18 = decltype(tmp17)::blendv(tmp15, tmp17, tmp3);
                    auto tmp20 = static_cast<float>(0.001);
                    auto tmp21 = at::vec::Vectorized<float>(tmp20);
                    auto tmp22 = tmp19 + tmp21;
                    auto tmp23 = tmp22.rsqrt();
                    auto tmp25 = tmp23 * tmp24;
                    auto tmp26 = tmp18 * tmp25;
                    tmp26.store(in_out_ptr1 + static_cast<long>(x1 + (16L*x0)));
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, convolution, clone, div, convolution_1, relu, convolution_2, add_7, convolution_3, relu_1, convolution_4, relu_2, convolution_5, add_13, convolution_6, relu_3, convolution_7, relu_4, convolution_8, add_20, convolution_9, relu_5, convolution_10, relu_6, mean, relu_7, div_1, mul_34, convolution_13, add_27, convolution_14, relu_8, convolution_15, relu_9, mean_1, relu_10, div_2, mul_44, convolution_18, add_35, convolution_19, relu_11, convolution_20, relu_12, mean_2, relu_13, div_3, mul_54, convolution_23, add_43, convolution_24, clone_1, div_4, convolution_25, clone_2, div_5, convolution_26, add_51, convolution_27, clone_3, div_6, convolution_28, clone_4, div_7, convolution_29, add_60, convolution_30, clone_5, div_8, convolution_31, clone_6, div_9, convolution_32, add_69, convolution_33, clone_7, div_10, convolution_34, clone_8, div_11, convolution_35, add_78, convolution_36, clone_9, div_12, convolution_37, clone_10, div_13, mean_3, relu_14, div_14, mul_110, convolution_40, add_87, convolution_41, clone_11, div_15, convolution_42, clone_12, div_16, mean_4, relu_15, div_17, mul_122, convolution_45, add_97, convolution_46, clone_13, div_18, convolution_47, clone_14, div_19, mean_5, relu_16, div_20, mul_134, convolution_50, add_106, convolution_51, clone_15, div_21, convolution_52, clone_16, div_22, mean_6, relu_17, div_23, mul_146, convolution_55, add_116, convolution_56, clone_17, div_24, convolution_57, clone_18, div_25, mean_7, relu_18, div_26, mul_158, convolution_60, add_126, convolution_61, clone_19, view, addmm, div_28, permute_2, permute_6, bitwise_and, bitwise_and_1, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 1, 9, 3))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_4, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_10, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_16, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (24, ), (1, ))
    assert_size_stride(primals_19, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_20, (72, ), (1, ))
    assert_size_stride(primals_22, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (72, ), (1, ))
    assert_size_stride(primals_25, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_28, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_29, (72, ), (1, ))
    assert_size_stride(primals_31, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_32, (72, ), (1, ))
    assert_size_stride(primals_34, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_36, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_38, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(primals_39, (40, ), (1, ))
    assert_size_stride(primals_41, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_42, (120, ), (1, ))
    assert_size_stride(primals_44, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_45, (120, ), (1, ))
    assert_size_stride(primals_47, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_49, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_51, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_52, (40, ), (1, ))
    assert_size_stride(primals_54, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_55, (120, ), (1, ))
    assert_size_stride(primals_57, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_58, (120, ), (1, ))
    assert_size_stride(primals_60, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_62, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_64, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_65, (40, ), (1, ))
    assert_size_stride(primals_67, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(primals_68, (240, ), (1, ))
    assert_size_stride(primals_70, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (240, ), (1, ))
    assert_size_stride(primals_73, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_74, (80, ), (1, ))
    assert_size_stride(primals_76, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_77, (200, ), (1, ))
    assert_size_stride(primals_79, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (200, ), (1, ))
    assert_size_stride(primals_82, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_83, (80, ), (1, ))
    assert_size_stride(primals_85, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_86, (184, ), (1, ))
    assert_size_stride(primals_88, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_89, (184, ), (1, ))
    assert_size_stride(primals_91, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_92, (80, ), (1, ))
    assert_size_stride(primals_94, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_95, (184, ), (1, ))
    assert_size_stride(primals_97, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (184, ), (1, ))
    assert_size_stride(primals_100, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(primals_101, (80, ), (1, ))
    assert_size_stride(primals_103, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_104, (480, ), (1, ))
    assert_size_stride(primals_106, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_107, (480, ), (1, ))
    assert_size_stride(primals_109, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_111, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(primals_113, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_114, (112, ), (1, ))
    assert_size_stride(primals_116, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_117, (672, ), (1, ))
    assert_size_stride(primals_119, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_120, (672, ), (1, ))
    assert_size_stride(primals_122, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_124, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_126, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_127, (112, ), (1, ))
    assert_size_stride(primals_129, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(primals_130, (672, ), (1, ))
    assert_size_stride(primals_132, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_133, (672, ), (1, ))
    assert_size_stride(primals_135, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_137, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(primals_139, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_142, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_143, (960, ), (1, ))
    assert_size_stride(primals_145, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_146, (960, ), (1, ))
    assert_size_stride(primals_148, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_150, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_152, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_153, (160, ), (1, ))
    assert_size_stride(primals_155, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_156, (960, ), (1, ))
    assert_size_stride(primals_158, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(primals_159, (960, ), (1, ))
    assert_size_stride(primals_161, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_163, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_165, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_166, (160, ), (1, ))
    assert_size_stride(primals_168, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_169, (960, ), (1, ))
    assert_size_stride(primals_175, (16, ), (1, ))
    assert_size_stride(primals_176, (16, ), (1, ))
    assert_size_stride(primals_178, (16, ), (1, ))
    assert_size_stride(primals_179, (16, ), (1, ))
    assert_size_stride(primals_181, (16, ), (1, ))
    assert_size_stride(primals_182, (16, ), (1, ))
    assert_size_stride(primals_184, (64, ), (1, ))
    assert_size_stride(primals_185, (64, ), (1, ))
    assert_size_stride(primals_187, (64, ), (1, ))
    assert_size_stride(primals_188, (64, ), (1, ))
    assert_size_stride(primals_190, (24, ), (1, ))
    assert_size_stride(primals_191, (24, ), (1, ))
    assert_size_stride(primals_193, (72, ), (1, ))
    assert_size_stride(primals_194, (72, ), (1, ))
    assert_size_stride(primals_196, (72, ), (1, ))
    assert_size_stride(primals_197, (72, ), (1, ))
    assert_size_stride(primals_199, (24, ), (1, ))
    assert_size_stride(primals_200, (24, ), (1, ))
    assert_size_stride(primals_202, (72, ), (1, ))
    assert_size_stride(primals_203, (72, ), (1, ))
    assert_size_stride(primals_205, (72, ), (1, ))
    assert_size_stride(primals_206, (72, ), (1, ))
    assert_size_stride(primals_208, (40, ), (1, ))
    assert_size_stride(primals_209, (40, ), (1, ))
    assert_size_stride(primals_211, (120, ), (1, ))
    assert_size_stride(primals_212, (120, ), (1, ))
    assert_size_stride(primals_214, (120, ), (1, ))
    assert_size_stride(primals_215, (120, ), (1, ))
    assert_size_stride(primals_217, (40, ), (1, ))
    assert_size_stride(primals_218, (40, ), (1, ))
    assert_size_stride(primals_220, (120, ), (1, ))
    assert_size_stride(primals_221, (120, ), (1, ))
    assert_size_stride(primals_223, (120, ), (1, ))
    assert_size_stride(primals_224, (120, ), (1, ))
    assert_size_stride(primals_226, (40, ), (1, ))
    assert_size_stride(primals_227, (40, ), (1, ))
    assert_size_stride(primals_229, (240, ), (1, ))
    assert_size_stride(primals_230, (240, ), (1, ))
    assert_size_stride(primals_232, (240, ), (1, ))
    assert_size_stride(primals_233, (240, ), (1, ))
    assert_size_stride(primals_235, (80, ), (1, ))
    assert_size_stride(primals_236, (80, ), (1, ))
    assert_size_stride(primals_238, (200, ), (1, ))
    assert_size_stride(primals_239, (200, ), (1, ))
    assert_size_stride(primals_241, (200, ), (1, ))
    assert_size_stride(primals_242, (200, ), (1, ))
    assert_size_stride(primals_244, (80, ), (1, ))
    assert_size_stride(primals_245, (80, ), (1, ))
    assert_size_stride(primals_247, (184, ), (1, ))
    assert_size_stride(primals_248, (184, ), (1, ))
    assert_size_stride(primals_250, (184, ), (1, ))
    assert_size_stride(primals_251, (184, ), (1, ))
    assert_size_stride(primals_253, (80, ), (1, ))
    assert_size_stride(primals_254, (80, ), (1, ))
    assert_size_stride(primals_256, (184, ), (1, ))
    assert_size_stride(primals_257, (184, ), (1, ))
    assert_size_stride(primals_259, (184, ), (1, ))
    assert_size_stride(primals_260, (184, ), (1, ))
    assert_size_stride(primals_262, (80, ), (1, ))
    assert_size_stride(primals_263, (80, ), (1, ))
    assert_size_stride(primals_265, (480, ), (1, ))
    assert_size_stride(primals_266, (480, ), (1, ))
    assert_size_stride(primals_268, (480, ), (1, ))
    assert_size_stride(primals_269, (480, ), (1, ))
    assert_size_stride(primals_271, (112, ), (1, ))
    assert_size_stride(primals_272, (112, ), (1, ))
    assert_size_stride(primals_274, (672, ), (1, ))
    assert_size_stride(primals_275, (672, ), (1, ))
    assert_size_stride(primals_277, (672, ), (1, ))
    assert_size_stride(primals_278, (672, ), (1, ))
    assert_size_stride(primals_280, (112, ), (1, ))
    assert_size_stride(primals_281, (112, ), (1, ))
    assert_size_stride(primals_283, (672, ), (1, ))
    assert_size_stride(primals_284, (672, ), (1, ))
    assert_size_stride(primals_286, (672, ), (1, ))
    assert_size_stride(primals_287, (672, ), (1, ))
    assert_size_stride(primals_289, (160, ), (1, ))
    assert_size_stride(primals_290, (160, ), (1, ))
    assert_size_stride(primals_292, (960, ), (1, ))
    assert_size_stride(primals_293, (960, ), (1, ))
    assert_size_stride(primals_295, (960, ), (1, ))
    assert_size_stride(primals_296, (960, ), (1, ))
    assert_size_stride(primals_298, (160, ), (1, ))
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_301, (960, ), (1, ))
    assert_size_stride(primals_302, (960, ), (1, ))
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_310, (960, ), (1, ))
    assert_size_stride(primals_311, (960, ), (1, ))
    assert_size_stride(primals_313, (4, 3, 224, 224), (150528, 1, 672, 3))
    assert_size_stride(convolution, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(clone, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(div, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_1, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(relu, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_2, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(add_7, (4, 16, 112, 112), (200704, 1, 1792, 16))
    assert_size_stride(convolution_3, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(relu_1, (4, 64, 112, 112), (802816, 1, 7168, 64))
    assert_size_stride(convolution_4, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(relu_2, (4, 64, 56, 56), (200704, 1, 3584, 64))
    assert_size_stride(convolution_5, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_13, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_6, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_3, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_7, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_4, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_8, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(add_20, (4, 24, 56, 56), (75264, 1, 1344, 24))
    assert_size_stride(convolution_9, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(relu_5, (4, 72, 56, 56), (225792, 1, 4032, 72))
    assert_size_stride(convolution_10, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(relu_6, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(mean, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(relu_7, (4, 24, 1, 1), (24, 1, 24, 24))
    assert_size_stride(div_1, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(mul_34, (4, 72, 28, 28), (56448, 1, 2016, 72))
    assert_size_stride(convolution_13, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_27, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_14, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_8, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_15, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_9, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_1, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_10, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_2, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_44, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_18, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_35, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_19, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_11, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_20, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(relu_12, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(mean_2, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(relu_13, (4, 32, 1, 1), (32, 1, 32, 32))
    assert_size_stride(div_3, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(mul_54, (4, 120, 28, 28), (94080, 1, 3360, 120))
    assert_size_stride(convolution_23, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(add_43, (4, 40, 28, 28), (31360, 1, 1120, 40))
    assert_size_stride(convolution_24, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(clone_1, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(div_4, (4, 240, 28, 28), (188160, 1, 6720, 240))
    assert_size_stride(convolution_25, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(clone_2, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(div_5, (4, 240, 14, 14), (47040, 1, 3360, 240))
    assert_size_stride(convolution_26, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_51, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_27, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(clone_3, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_6, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_28, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(clone_4, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(div_7, (4, 200, 14, 14), (39200, 1, 2800, 200))
    assert_size_stride(convolution_29, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_60, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_30, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_5, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_8, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_31, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_6, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_9, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_32, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_69, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_33, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_7, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_10, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_34, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(clone_8, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(div_11, (4, 184, 14, 14), (36064, 1, 2576, 184))
    assert_size_stride(convolution_35, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(add_78, (4, 80, 14, 14), (15680, 1, 1120, 80))
    assert_size_stride(convolution_36, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(clone_9, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_12, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_37, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(clone_10, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(div_13, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(mean_3, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(relu_14, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(div_14, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(mul_110, (4, 480, 14, 14), (94080, 1, 6720, 480))
    assert_size_stride(convolution_40, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_87, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_41, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_11, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_15, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_42, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_12, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_16, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(mean_4, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_15, (4, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_17, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_122, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_45, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(add_97, (4, 112, 14, 14), (21952, 1, 1568, 112))
    assert_size_stride(convolution_46, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(clone_13, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(div_18, (4, 672, 14, 14), (131712, 1, 9408, 672))
    assert_size_stride(convolution_47, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(clone_14, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(div_19, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(mean_5, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(relu_16, (4, 168, 1, 1), (168, 1, 168, 168))
    assert_size_stride(div_20, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(mul_134, (4, 672, 7, 7), (32928, 1, 4704, 672))
    assert_size_stride(convolution_50, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_106, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_51, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_15, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_21, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_52, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_16, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_22, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_6, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_17, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_23, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_146, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_55, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_116, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_56, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_17, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_24, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_57, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_18, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(div_25, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(mean_7, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(relu_18, (4, 240, 1, 1), (240, 1, 240, 240))
    assert_size_stride(div_26, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(mul_158, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(convolution_60, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(add_126, (4, 160, 7, 7), (7840, 1, 1120, 160))
    assert_size_stride(convolution_61, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(clone_19, (4, 960, 7, 7), (47040, 1, 6720, 960))
    assert_size_stride(view, (4, 960), (960, 1))
    assert_size_stride(addmm, (4, 1280), (1280, 1))
    assert_size_stride(div_28, (4, 1280), (1280, 1))
    assert_size_stride(permute_2, (1000, 1280), (1280, 1))
    assert_size_stride(permute_6, (1280, 960), (960, 1))
    assert_size_stride(bitwise_and, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(bitwise_and_1, (4, 960, 1, 1), (960, 1, 960, 960))
    assert_size_stride(bitwise_and_2, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(bitwise_and_3, (4, 672, 1, 1), (672, 1, 672, 672))
    assert_size_stride(bitwise_and_4, (4, 480, 1, 1), (480, 1, 480, 480))
    assert_size_stride(bitwise_and_5, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(bitwise_and_6, (4, 120, 1, 1), (120, 1, 120, 120))
    assert_size_stride(bitwise_and_7, (4, 72, 1, 1), (72, 1, 72, 72))
    assert_size_stride(tangents_1, (4, 1000), (1000, 1))
    buf0 = empty((4, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(tangents_1, permute_2, out=buf0)
    del permute_2
    buf1 = empty((1000, 1280), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(tangents_1, (1000, 4), (1, 1000), 0), div_28, out=buf1)
    del div_28
    buf2 = empty((1000, ), device='cpu', dtype=torch.float32)
    buf3 = buf0; del buf0  # reuse
    cpp_fused_hardswish_backward_sum_view_0(c_void_p(buf3.data_ptr()), c_void_p(tangents_1.data_ptr()), c_void_p(addmm.data_ptr()), c_void_p(buf2.data_ptr()))
    del addmm
    del tangents_1
    buf4 = empty((4, 960), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(buf3, permute_6, out=buf4)
    del permute_6
    buf5 = empty((1280, 960), device='cpu', dtype=torch.float32)
    # Source Nodes: [], Original ATen: [aten.mm]
    extern_kernels.mm(reinterpret_tensor(buf3, (1280, 4), (1, 1280), 0), view, out=buf5)
    del view
    buf6 = empty((1280, ), device='cpu', dtype=torch.float32)
    buf7 = empty((960, ), device='cpu', dtype=torch.float32)
    buf8 = empty((960, ), device='cpu', dtype=torch.float32)
    buf9 = buf8; del buf8  # reuse
    buf10 = empty_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_div_hardswish_backward_native_batch_norm_backward_sum_view_1(c_void_p(buf9.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(clone_19.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(convolution_61.data_ptr()), c_void_p(primals_310.data_ptr()), c_void_p(primals_311.data_ptr()), c_void_p(primals_169.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf10.data_ptr()))
    del buf3
    del clone_19
    del convolution_61
    del primals_169
    del primals_310
    del primals_311
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.div, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf11 = aten.convolution_backward(buf10, add_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_126
    del buf10
    del primals_168
    buf12 = buf11[0]
    buf13 = buf11[1]
    del buf11
    buf14 = empty((160, ), device='cpu', dtype=torch.float32)
    buf15 = empty((160, ), device='cpu', dtype=torch.float32)
    buf16 = buf15; del buf15  # reuse
    buf17 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_2(c_void_p(buf16.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(convolution_60.data_ptr()), c_void_p(primals_307.data_ptr()), c_void_p(primals_308.data_ptr()), c_void_p(primals_166.data_ptr()), c_void_p(buf14.data_ptr()), c_void_p(buf17.data_ptr()))
    del convolution_60
    del primals_166
    del primals_307
    del primals_308
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf18 = aten.convolution_backward(buf17, mul_158, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_158
    del primals_165
    buf19 = buf18[0]
    buf20 = buf18[1]
    del buf18
    buf21 = reinterpret_tensor(buf4, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf4  # reuse
    buf22 = reinterpret_tensor(buf21, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf21  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_3(c_void_p(buf22.data_ptr()), c_void_p(buf19.data_ptr()), c_void_p(div_25.data_ptr()), c_void_p(bitwise_and.data_ptr()))
    del bitwise_and
    del div_25
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf23 = aten.convolution_backward(buf22, relu_18, primals_163, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf22
    del primals_163
    buf24 = buf23[0]
    buf25 = buf23[1]
    buf26 = buf23[2]
    del buf23
    buf27 = buf24; del buf24  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_4(c_void_p(buf27.data_ptr()), c_void_p(relu_18.data_ptr()))
    del relu_18
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf28 = aten.convolution_backward(buf27, mean_7, primals_161, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_7
    del primals_161
    buf29 = buf28[0]
    buf30 = buf28[1]
    buf31 = buf28[2]
    del buf28
    buf32 = reinterpret_tensor(buf27, (960, ), (1, ), 0); del buf27  # reuse
    buf33 = empty((960, ), device='cpu', dtype=torch.float32)
    buf34 = buf33; del buf33  # reuse
    buf35 = buf19; del buf19  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_5(c_void_p(buf34.data_ptr()), c_void_p(buf35.data_ptr()), c_void_p(clone_18.data_ptr()), c_void_p(div_26.data_ptr()), c_void_p(buf29.data_ptr()), c_void_p(convolution_57.data_ptr()), c_void_p(primals_304.data_ptr()), c_void_p(primals_305.data_ptr()), c_void_p(primals_159.data_ptr()), c_void_p(buf32.data_ptr()))
    del clone_18
    del convolution_57
    del div_26
    del primals_159
    del primals_304
    del primals_305
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
    buf36 = aten.convolution_backward(buf35, div_24, primals_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
    del buf35
    del div_24
    del primals_158
    buf37 = buf36[0]
    buf38 = buf36[1]
    del buf36
    buf39 = empty((960, ), device='cpu', dtype=torch.float32)
    buf40 = empty((960, ), device='cpu', dtype=torch.float32)
    buf41 = buf40; del buf40  # reuse
    buf42 = buf37; del buf37  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_6(c_void_p(buf41.data_ptr()), c_void_p(buf42.data_ptr()), c_void_p(clone_17.data_ptr()), c_void_p(convolution_56.data_ptr()), c_void_p(primals_301.data_ptr()), c_void_p(primals_302.data_ptr()), c_void_p(primals_156.data_ptr()), c_void_p(buf39.data_ptr()))
    del clone_17
    del convolution_56
    del primals_156
    del primals_301
    del primals_302
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf43 = aten.convolution_backward(buf42, add_116, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_116
    del primals_155
    buf44 = buf43[0]
    buf45 = buf43[1]
    del buf43
    buf49 = buf17; del buf17  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_7(c_void_p(buf12.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(primals_153.data_ptr()), c_void_p(buf49.data_ptr()))
    del primals_153
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf50 = aten.convolution_backward(buf49, mul_146, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf49
    del mul_146
    del primals_152
    buf51 = buf50[0]
    buf53 = reinterpret_tensor(buf29, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf29  # reuse
    buf54 = reinterpret_tensor(buf53, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf53  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_8(c_void_p(buf54.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(div_22.data_ptr()), c_void_p(bitwise_and_1.data_ptr()))
    del bitwise_and_1
    del div_22
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf55 = aten.convolution_backward(buf54, relu_17, primals_150, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf54
    del primals_150
    buf56 = buf55[0]
    buf59 = buf56; del buf56  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_9(c_void_p(buf59.data_ptr()), c_void_p(relu_17.data_ptr()))
    del relu_17
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf60 = aten.convolution_backward(buf59, mean_6, primals_148, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_6
    del primals_148
    buf61 = buf60[0]
    buf67 = buf42; del buf42  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_10(c_void_p(clone_16.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(primals_146.data_ptr()), c_void_p(buf67.data_ptr()))
    del primals_146
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
    buf68 = aten.convolution_backward(buf67, div_21, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False])
    del div_21
    del primals_145
    buf69 = buf68[0]
    buf74 = buf67; del buf67  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_11(c_void_p(clone_15.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(primals_143.data_ptr()), c_void_p(buf74.data_ptr()))
    del primals_143
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf75 = aten.convolution_backward(buf74, add_106, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_106
    del buf74
    del primals_142
    buf76 = buf75[0]
    buf46 = empty((160, ), device='cpu', dtype=torch.float32)
    buf47 = empty((160, ), device='cpu', dtype=torch.float32)
    buf78 = empty((160, ), device='cpu', dtype=torch.float32)
    buf79 = empty((160, ), device='cpu', dtype=torch.float32)
    buf48 = buf47; del buf47  # reuse
    cpp_fused_add_native_batch_norm_backward_12(c_void_p(buf48.data_ptr()), c_void_p(buf12.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(convolution_55.data_ptr()), c_void_p(primals_298.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(convolution_50.data_ptr()), c_void_p(primals_289.data_ptr()), c_void_p(primals_299.data_ptr()), c_void_p(buf46.data_ptr()), c_void_p(buf78.data_ptr()), c_void_p(buf79.data_ptr()))
    del convolution_50
    del convolution_55
    del primals_289
    del primals_298
    del primals_299
    buf52 = buf50[1]
    del buf50
    buf57 = buf55[1]
    buf58 = buf55[2]
    del buf55
    buf62 = buf60[1]
    buf63 = buf60[2]
    del buf60
    buf64 = reinterpret_tensor(buf59, (960, ), (1, ), 0); del buf59  # reuse
    buf65 = empty((960, ), device='cpu', dtype=torch.float32)
    buf66 = buf65; del buf65  # reuse
    cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_13(c_void_p(buf66.data_ptr()), c_void_p(clone_16.data_ptr()), c_void_p(buf51.data_ptr()), c_void_p(div_23.data_ptr()), c_void_p(buf61.data_ptr()), c_void_p(convolution_52.data_ptr()), c_void_p(primals_295.data_ptr()), c_void_p(primals_296.data_ptr()), c_void_p(buf64.data_ptr()))
    del buf51
    del buf61
    del clone_16
    del convolution_52
    del div_23
    del primals_295
    del primals_296
    buf70 = buf68[1]
    del buf68
    buf71 = empty((960, ), device='cpu', dtype=torch.float32)
    buf72 = empty((960, ), device='cpu', dtype=torch.float32)
    buf73 = buf72; del buf72  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_14(c_void_p(buf73.data_ptr()), c_void_p(clone_15.data_ptr()), c_void_p(buf69.data_ptr()), c_void_p(convolution_51.data_ptr()), c_void_p(primals_292.data_ptr()), c_void_p(primals_293.data_ptr()), c_void_p(buf71.data_ptr()))
    del buf69
    del clone_15
    del convolution_51
    del primals_292
    del primals_293
    buf77 = buf75[1]
    del buf75
    buf80 = buf79; del buf79  # reuse
    buf81 = buf12; del buf12  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_15(c_void_p(buf80.data_ptr()), c_void_p(buf81.data_ptr()), c_void_p(primals_290.data_ptr()), c_void_p(buf44.data_ptr()), c_void_p(buf76.data_ptr()), c_void_p(primals_140.data_ptr()))
    del buf44
    del buf76
    del primals_140
    del primals_290
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf82 = aten.convolution_backward(buf81, mul_134, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf81
    del mul_134
    del primals_139
    buf83 = buf82[0]
    buf84 = buf82[1]
    del buf82
    buf85 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cpu', dtype=torch.float32)
    buf86 = reinterpret_tensor(buf85, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf85  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_16(c_void_p(buf86.data_ptr()), c_void_p(buf83.data_ptr()), c_void_p(div_19.data_ptr()), c_void_p(bitwise_and_2.data_ptr()))
    del bitwise_and_2
    del div_19
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf87 = aten.convolution_backward(buf86, relu_16, primals_137, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf86
    del primals_137
    buf88 = buf87[0]
    buf89 = buf87[1]
    buf90 = buf87[2]
    del buf87
    buf91 = buf88; del buf88  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_17(c_void_p(buf91.data_ptr()), c_void_p(relu_16.data_ptr()))
    del relu_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf92 = aten.convolution_backward(buf91, mean_5, primals_135, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_5
    del primals_135
    buf93 = buf92[0]
    buf94 = buf92[1]
    buf95 = buf92[2]
    del buf92
    buf96 = reinterpret_tensor(buf91, (672, ), (1, ), 0); del buf91  # reuse
    buf97 = empty((672, ), device='cpu', dtype=torch.float32)
    buf98 = buf97; del buf97  # reuse
    buf99 = buf83; del buf83  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_18(c_void_p(buf98.data_ptr()), c_void_p(buf99.data_ptr()), c_void_p(clone_14.data_ptr()), c_void_p(div_20.data_ptr()), c_void_p(buf93.data_ptr()), c_void_p(convolution_47.data_ptr()), c_void_p(primals_286.data_ptr()), c_void_p(primals_287.data_ptr()), c_void_p(primals_133.data_ptr()), c_void_p(buf96.data_ptr()))
    del clone_14
    del convolution_47
    del div_20
    del primals_133
    del primals_286
    del primals_287
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
    buf100 = aten.convolution_backward(buf99, div_18, primals_132, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False])
    del buf99
    del div_18
    del primals_132
    buf101 = buf100[0]
    buf102 = buf100[1]
    del buf100
    buf103 = empty((672, ), device='cpu', dtype=torch.float32)
    buf104 = empty((672, ), device='cpu', dtype=torch.float32)
    buf105 = buf104; del buf104  # reuse
    buf106 = buf101; del buf101  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_19(c_void_p(buf105.data_ptr()), c_void_p(buf106.data_ptr()), c_void_p(clone_13.data_ptr()), c_void_p(convolution_46.data_ptr()), c_void_p(primals_283.data_ptr()), c_void_p(primals_284.data_ptr()), c_void_p(primals_130.data_ptr()), c_void_p(buf103.data_ptr()))
    del clone_13
    del convolution_46
    del primals_130
    del primals_283
    del primals_284
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf107 = aten.convolution_backward(buf106, add_97, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_97
    del primals_129
    buf108 = buf107[0]
    buf109 = buf107[1]
    del buf107
    buf113 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_20(c_void_p(buf108.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(primals_127.data_ptr()), c_void_p(buf113.data_ptr()))
    del primals_127
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf114 = aten.convolution_backward(buf113, mul_122, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf113
    del mul_122
    del primals_126
    buf115 = buf114[0]
    buf117 = reinterpret_tensor(buf93, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf93  # reuse
    buf118 = reinterpret_tensor(buf117, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf117  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_21(c_void_p(buf118.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(div_16.data_ptr()), c_void_p(bitwise_and_3.data_ptr()))
    del bitwise_and_3
    del div_16
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf119 = aten.convolution_backward(buf118, relu_15, primals_124, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf118
    del primals_124
    buf120 = buf119[0]
    buf123 = buf120; del buf120  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_22(c_void_p(buf123.data_ptr()), c_void_p(relu_15.data_ptr()))
    del relu_15
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf124 = aten.convolution_backward(buf123, mean_4, primals_122, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_4
    del primals_122
    buf125 = buf124[0]
    buf131 = buf106; del buf106  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_23(c_void_p(clone_12.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(primals_120.data_ptr()), c_void_p(buf131.data_ptr()))
    del primals_120
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
    buf132 = aten.convolution_backward(buf131, div_15, primals_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False])
    del div_15
    del primals_119
    buf133 = buf132[0]
    buf138 = buf131; del buf131  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_24(c_void_p(clone_11.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(primals_117.data_ptr()), c_void_p(buf138.data_ptr()))
    del primals_117
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf139 = aten.convolution_backward(buf138, add_87, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_87
    del buf138
    del primals_116
    buf140 = buf139[0]
    buf110 = empty((112, ), device='cpu', dtype=torch.float32)
    buf111 = empty((112, ), device='cpu', dtype=torch.float32)
    buf142 = empty((112, ), device='cpu', dtype=torch.float32)
    buf143 = empty((112, ), device='cpu', dtype=torch.float32)
    buf112 = buf111; del buf111  # reuse
    cpp_fused_add_native_batch_norm_backward_25(c_void_p(buf112.data_ptr()), c_void_p(buf108.data_ptr()), c_void_p(convolution_45.data_ptr()), c_void_p(primals_280.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(convolution_40.data_ptr()), c_void_p(primals_271.data_ptr()), c_void_p(primals_281.data_ptr()), c_void_p(buf110.data_ptr()), c_void_p(buf142.data_ptr()), c_void_p(buf143.data_ptr()))
    del convolution_40
    del convolution_45
    del primals_271
    del primals_280
    del primals_281
    buf116 = buf114[1]
    del buf114
    buf121 = buf119[1]
    buf122 = buf119[2]
    del buf119
    buf126 = buf124[1]
    buf127 = buf124[2]
    del buf124
    buf128 = reinterpret_tensor(buf123, (672, ), (1, ), 0); del buf123  # reuse
    buf129 = empty((672, ), device='cpu', dtype=torch.float32)
    buf130 = buf129; del buf129  # reuse
    cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_26(c_void_p(buf130.data_ptr()), c_void_p(clone_12.data_ptr()), c_void_p(buf115.data_ptr()), c_void_p(div_17.data_ptr()), c_void_p(buf125.data_ptr()), c_void_p(convolution_42.data_ptr()), c_void_p(primals_277.data_ptr()), c_void_p(primals_278.data_ptr()), c_void_p(buf128.data_ptr()))
    del buf115
    del buf125
    del clone_12
    del convolution_42
    del div_17
    del primals_277
    del primals_278
    buf134 = buf132[1]
    del buf132
    buf135 = empty((672, ), device='cpu', dtype=torch.float32)
    buf136 = empty((672, ), device='cpu', dtype=torch.float32)
    buf137 = buf136; del buf136  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_27(c_void_p(buf137.data_ptr()), c_void_p(clone_11.data_ptr()), c_void_p(buf133.data_ptr()), c_void_p(convolution_41.data_ptr()), c_void_p(primals_274.data_ptr()), c_void_p(primals_275.data_ptr()), c_void_p(buf135.data_ptr()))
    del buf133
    del clone_11
    del convolution_41
    del primals_274
    del primals_275
    buf141 = buf139[1]
    del buf139
    buf144 = buf143; del buf143  # reuse
    buf145 = buf108; del buf108  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_28(c_void_p(buf144.data_ptr()), c_void_p(buf145.data_ptr()), c_void_p(primals_272.data_ptr()), c_void_p(buf140.data_ptr()), c_void_p(primals_114.data_ptr()))
    del buf140
    del primals_114
    del primals_272
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf146 = aten.convolution_backward(buf145, mul_110, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf145
    del mul_110
    del primals_113
    buf147 = buf146[0]
    buf148 = buf146[1]
    del buf146
    buf149 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cpu', dtype=torch.float32)
    buf150 = reinterpret_tensor(buf149, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf149  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_29(c_void_p(buf150.data_ptr()), c_void_p(buf147.data_ptr()), c_void_p(div_13.data_ptr()), c_void_p(bitwise_and_4.data_ptr()))
    del bitwise_and_4
    del div_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf151 = aten.convolution_backward(buf150, relu_14, primals_111, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf150
    del primals_111
    buf152 = buf151[0]
    buf153 = buf151[1]
    buf154 = buf151[2]
    del buf151
    buf155 = buf152; del buf152  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_30(c_void_p(buf155.data_ptr()), c_void_p(relu_14.data_ptr()))
    del relu_14
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf156 = aten.convolution_backward(buf155, mean_3, primals_109, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del mean_3
    del primals_109
    buf157 = buf156[0]
    buf158 = buf156[1]
    buf159 = buf156[2]
    del buf156
    buf160 = reinterpret_tensor(buf155, (480, ), (1, ), 0); del buf155  # reuse
    buf161 = empty((480, ), device='cpu', dtype=torch.float32)
    buf162 = buf161; del buf161  # reuse
    buf163 = buf147; del buf147  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_31(c_void_p(buf162.data_ptr()), c_void_p(buf163.data_ptr()), c_void_p(clone_10.data_ptr()), c_void_p(div_14.data_ptr()), c_void_p(buf157.data_ptr()), c_void_p(convolution_37.data_ptr()), c_void_p(primals_268.data_ptr()), c_void_p(primals_269.data_ptr()), c_void_p(primals_107.data_ptr()), c_void_p(buf160.data_ptr()))
    del buf157
    del clone_10
    del convolution_37
    del div_14
    del primals_107
    del primals_268
    del primals_269
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward]
    buf164 = aten.convolution_backward(buf163, div_12, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False])
    del buf163
    del div_12
    del primals_106
    buf165 = buf164[0]
    buf166 = buf164[1]
    del buf164
    buf167 = empty((480, ), device='cpu', dtype=torch.float32)
    buf168 = empty((480, ), device='cpu', dtype=torch.float32)
    buf169 = buf168; del buf168  # reuse
    buf170 = buf165; del buf165  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_32(c_void_p(buf169.data_ptr()), c_void_p(buf170.data_ptr()), c_void_p(clone_9.data_ptr()), c_void_p(convolution_36.data_ptr()), c_void_p(primals_265.data_ptr()), c_void_p(primals_266.data_ptr()), c_void_p(primals_104.data_ptr()), c_void_p(buf167.data_ptr()))
    del clone_9
    del convolution_36
    del primals_104
    del primals_265
    del primals_266
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf171 = aten.convolution_backward(buf170, add_78, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_78
    del buf170
    del primals_103
    buf172 = buf171[0]
    buf173 = buf171[1]
    del buf171
    buf177 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_33(c_void_p(buf172.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(primals_101.data_ptr()), c_void_p(buf177.data_ptr()))
    del primals_101
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf178 = aten.convolution_backward(buf177, div_11, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_11
    del primals_100
    buf179 = buf178[0]
    buf184 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_34(c_void_p(clone_8.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(primals_98.data_ptr()), c_void_p(buf184.data_ptr()))
    del primals_98
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf185 = aten.convolution_backward(buf184, div_10, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
    del div_10
    del primals_97
    buf186 = buf185[0]
    buf191 = buf184; del buf184  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_35(c_void_p(clone_7.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(primals_95.data_ptr()), c_void_p(buf191.data_ptr()))
    del primals_95
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf192 = aten.convolution_backward(buf191, add_69, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_69
    del primals_94
    buf193 = buf192[0]
    buf198 = buf177; del buf177  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_36(c_void_p(buf172.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(primals_254.data_ptr()), c_void_p(primals_92.data_ptr()), c_void_p(buf198.data_ptr()))
    del primals_92
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf199 = aten.convolution_backward(buf198, div_9, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del div_9
    del primals_91
    buf200 = buf199[0]
    buf205 = buf191; del buf191  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_37(c_void_p(clone_6.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(primals_89.data_ptr()), c_void_p(buf205.data_ptr()))
    del primals_89
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf206 = aten.convolution_backward(buf205, div_8, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False])
    del div_8
    del primals_88
    buf207 = buf206[0]
    buf212 = buf205; del buf205  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_38(c_void_p(clone_5.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(primals_86.data_ptr()), c_void_p(buf212.data_ptr()))
    del primals_86
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf213 = aten.convolution_backward(buf212, add_60, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_60
    del buf212
    del primals_85
    buf214 = buf213[0]
    buf219 = buf198; del buf198  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_39(c_void_p(buf172.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(primals_245.data_ptr()), c_void_p(primals_83.data_ptr()), c_void_p(buf219.data_ptr()))
    del primals_83
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf220 = aten.convolution_backward(buf219, div_7, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf219
    del div_7
    del primals_82
    buf221 = buf220[0]
    buf226 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_40(c_void_p(clone_4.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(primals_80.data_ptr()), c_void_p(buf226.data_ptr()))
    del primals_80
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf227 = aten.convolution_backward(buf226, div_6, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False])
    del div_6
    del primals_79
    buf228 = buf227[0]
    buf233 = buf226; del buf226  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_41(c_void_p(clone_3.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(primals_77.data_ptr()), c_void_p(buf233.data_ptr()))
    del primals_77
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf234 = aten.convolution_backward(buf233, add_51, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_51
    del buf233
    del primals_76
    buf235 = buf234[0]
    buf174 = empty((80, ), device='cpu', dtype=torch.float32)
    buf175 = empty((80, ), device='cpu', dtype=torch.float32)
    buf195 = empty((80, ), device='cpu', dtype=torch.float32)
    buf196 = empty((80, ), device='cpu', dtype=torch.float32)
    buf216 = empty((80, ), device='cpu', dtype=torch.float32)
    buf217 = empty((80, ), device='cpu', dtype=torch.float32)
    buf237 = empty((80, ), device='cpu', dtype=torch.float32)
    buf238 = empty((80, ), device='cpu', dtype=torch.float32)
    buf176 = buf175; del buf175  # reuse
    cpp_fused_add_native_batch_norm_backward_42(c_void_p(buf176.data_ptr()), c_void_p(buf172.data_ptr()), c_void_p(convolution_35.data_ptr()), c_void_p(primals_262.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(convolution_32.data_ptr()), c_void_p(primals_253.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(convolution_29.data_ptr()), c_void_p(primals_244.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(convolution_26.data_ptr()), c_void_p(primals_235.data_ptr()), c_void_p(primals_263.data_ptr()), c_void_p(buf174.data_ptr()), c_void_p(buf195.data_ptr()), c_void_p(buf196.data_ptr()), c_void_p(buf216.data_ptr()), c_void_p(buf217.data_ptr()), c_void_p(buf237.data_ptr()), c_void_p(buf238.data_ptr()))
    del convolution_26
    del convolution_29
    del convolution_32
    del convolution_35
    del primals_235
    del primals_244
    del primals_253
    del primals_262
    del primals_263
    buf180 = buf178[1]
    del buf178
    buf181 = empty((184, ), device='cpu', dtype=torch.float32)
    buf182 = empty((184, ), device='cpu', dtype=torch.float32)
    buf183 = buf182; del buf182  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_43(c_void_p(buf183.data_ptr()), c_void_p(clone_8.data_ptr()), c_void_p(buf179.data_ptr()), c_void_p(convolution_34.data_ptr()), c_void_p(primals_259.data_ptr()), c_void_p(primals_260.data_ptr()), c_void_p(buf181.data_ptr()))
    del buf179
    del clone_8
    del convolution_34
    del primals_259
    del primals_260
    buf187 = buf185[1]
    del buf185
    buf188 = empty((184, ), device='cpu', dtype=torch.float32)
    buf189 = empty((184, ), device='cpu', dtype=torch.float32)
    buf190 = buf189; del buf189  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_44(c_void_p(buf190.data_ptr()), c_void_p(clone_7.data_ptr()), c_void_p(buf186.data_ptr()), c_void_p(convolution_33.data_ptr()), c_void_p(primals_256.data_ptr()), c_void_p(primals_257.data_ptr()), c_void_p(buf188.data_ptr()))
    del buf186
    del clone_7
    del convolution_33
    del primals_256
    del primals_257
    buf194 = buf192[1]
    del buf192
    buf197 = buf196; del buf196  # reuse
    cpp_fused_native_batch_norm_backward_45(c_void_p(buf197.data_ptr()), c_void_p(primals_254.data_ptr()))
    del primals_254
    buf201 = buf199[1]
    del buf199
    buf202 = empty((184, ), device='cpu', dtype=torch.float32)
    buf203 = empty((184, ), device='cpu', dtype=torch.float32)
    buf204 = buf203; del buf203  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_46(c_void_p(buf204.data_ptr()), c_void_p(clone_6.data_ptr()), c_void_p(buf200.data_ptr()), c_void_p(convolution_31.data_ptr()), c_void_p(primals_250.data_ptr()), c_void_p(primals_251.data_ptr()), c_void_p(buf202.data_ptr()))
    del buf200
    del clone_6
    del convolution_31
    del primals_250
    del primals_251
    buf208 = buf206[1]
    del buf206
    buf209 = empty((184, ), device='cpu', dtype=torch.float32)
    buf210 = empty((184, ), device='cpu', dtype=torch.float32)
    buf211 = buf210; del buf210  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_47(c_void_p(buf211.data_ptr()), c_void_p(clone_5.data_ptr()), c_void_p(buf207.data_ptr()), c_void_p(convolution_30.data_ptr()), c_void_p(primals_247.data_ptr()), c_void_p(primals_248.data_ptr()), c_void_p(buf209.data_ptr()))
    del buf207
    del clone_5
    del convolution_30
    del primals_247
    del primals_248
    buf215 = buf213[1]
    del buf213
    buf218 = buf217; del buf217  # reuse
    cpp_fused_native_batch_norm_backward_48(c_void_p(buf218.data_ptr()), c_void_p(primals_245.data_ptr()))
    del primals_245
    buf222 = buf220[1]
    del buf220
    buf223 = empty((200, ), device='cpu', dtype=torch.float32)
    buf224 = empty((200, ), device='cpu', dtype=torch.float32)
    buf225 = buf224; del buf224  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_49(c_void_p(buf225.data_ptr()), c_void_p(clone_4.data_ptr()), c_void_p(buf221.data_ptr()), c_void_p(convolution_28.data_ptr()), c_void_p(primals_241.data_ptr()), c_void_p(primals_242.data_ptr()), c_void_p(buf223.data_ptr()))
    del buf221
    del clone_4
    del convolution_28
    del primals_241
    del primals_242
    buf229 = buf227[1]
    del buf227
    buf230 = empty((200, ), device='cpu', dtype=torch.float32)
    buf231 = empty((200, ), device='cpu', dtype=torch.float32)
    buf232 = buf231; del buf231  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_50(c_void_p(buf232.data_ptr()), c_void_p(clone_3.data_ptr()), c_void_p(buf228.data_ptr()), c_void_p(convolution_27.data_ptr()), c_void_p(primals_238.data_ptr()), c_void_p(primals_239.data_ptr()), c_void_p(buf230.data_ptr()))
    del buf228
    del clone_3
    del convolution_27
    del primals_238
    del primals_239
    buf236 = buf234[1]
    del buf234
    buf239 = buf238; del buf238  # reuse
    buf240 = buf172; del buf172  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_51(c_void_p(buf239.data_ptr()), c_void_p(buf240.data_ptr()), c_void_p(primals_236.data_ptr()), c_void_p(buf193.data_ptr()), c_void_p(buf214.data_ptr()), c_void_p(buf235.data_ptr()), c_void_p(primals_74.data_ptr()))
    del buf193
    del buf214
    del buf235
    del primals_236
    del primals_74
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf241 = aten.convolution_backward(buf240, div_5, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf240
    del div_5
    del primals_73
    buf242 = buf241[0]
    buf243 = buf241[1]
    del buf241
    buf244 = empty((240, ), device='cpu', dtype=torch.float32)
    buf245 = empty((240, ), device='cpu', dtype=torch.float32)
    buf246 = buf245; del buf245  # reuse
    buf247 = buf242; del buf242  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_52(c_void_p(buf246.data_ptr()), c_void_p(buf247.data_ptr()), c_void_p(clone_2.data_ptr()), c_void_p(convolution_25.data_ptr()), c_void_p(primals_232.data_ptr()), c_void_p(primals_233.data_ptr()), c_void_p(primals_71.data_ptr()), c_void_p(buf244.data_ptr()))
    del clone_2
    del convolution_25
    del primals_232
    del primals_233
    del primals_71
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf248 = aten.convolution_backward(buf247, div_4, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False])
    del buf247
    del div_4
    del primals_70
    buf249 = buf248[0]
    buf250 = buf248[1]
    del buf248
    buf251 = empty((240, ), device='cpu', dtype=torch.float32)
    buf252 = empty((240, ), device='cpu', dtype=torch.float32)
    buf253 = buf252; del buf252  # reuse
    buf254 = buf249; del buf249  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_53(c_void_p(buf253.data_ptr()), c_void_p(buf254.data_ptr()), c_void_p(clone_1.data_ptr()), c_void_p(convolution_24.data_ptr()), c_void_p(primals_229.data_ptr()), c_void_p(primals_230.data_ptr()), c_void_p(primals_68.data_ptr()), c_void_p(buf251.data_ptr()))
    del clone_1
    del convolution_24
    del primals_229
    del primals_230
    del primals_68
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf255 = aten.convolution_backward(buf254, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_43
    del buf254
    del primals_67
    buf256 = buf255[0]
    buf257 = buf255[1]
    del buf255
    buf258 = empty((40, ), device='cpu', dtype=torch.float32)
    buf259 = empty((40, ), device='cpu', dtype=torch.float32)
    buf260 = buf259; del buf259  # reuse
    buf261 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_54(c_void_p(buf260.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(convolution_23.data_ptr()), c_void_p(primals_226.data_ptr()), c_void_p(primals_227.data_ptr()), c_void_p(primals_65.data_ptr()), c_void_p(buf258.data_ptr()), c_void_p(buf261.data_ptr()))
    del convolution_23
    del primals_226
    del primals_227
    del primals_65
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf262 = aten.convolution_backward(buf261, mul_54, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del mul_54
    del primals_64
    buf263 = buf262[0]
    buf264 = buf262[1]
    del buf262
    buf265 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cpu', dtype=torch.float32)
    buf266 = reinterpret_tensor(buf265, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf265  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_55(c_void_p(buf266.data_ptr()), c_void_p(buf263.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(bitwise_and_5.data_ptr()))
    del bitwise_and_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf267 = aten.convolution_backward(buf266, relu_13, primals_62, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf266
    del primals_62
    buf268 = buf267[0]
    buf269 = buf267[1]
    buf270 = buf267[2]
    del buf267
    buf271 = buf268; del buf268  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_56(c_void_p(buf271.data_ptr()), c_void_p(relu_13.data_ptr()))
    del relu_13
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf272 = aten.convolution_backward(buf271, mean_2, primals_60, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf271
    del mean_2
    del primals_60
    buf273 = buf272[0]
    buf274 = buf272[1]
    buf275 = buf272[2]
    del buf272
    buf276 = empty((120, ), device='cpu', dtype=torch.float32)
    buf277 = empty((120, ), device='cpu', dtype=torch.float32)
    buf278 = buf277; del buf277  # reuse
    buf279 = buf263; del buf263  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_57(c_void_p(buf278.data_ptr()), c_void_p(buf279.data_ptr()), c_void_p(relu_12.data_ptr()), c_void_p(div_3.data_ptr()), c_void_p(buf273.data_ptr()), c_void_p(convolution_20.data_ptr()), c_void_p(primals_223.data_ptr()), c_void_p(primals_224.data_ptr()), c_void_p(primals_58.data_ptr()), c_void_p(buf276.data_ptr()))
    del convolution_20
    del div_3
    del primals_223
    del primals_224
    del primals_58
    del relu_12
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
    buf280 = aten.convolution_backward(buf279, relu_11, primals_57, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del buf279
    del primals_57
    buf281 = buf280[0]
    buf282 = buf280[1]
    del buf280
    buf283 = empty((120, ), device='cpu', dtype=torch.float32)
    buf284 = empty((120, ), device='cpu', dtype=torch.float32)
    buf285 = buf284; del buf284  # reuse
    buf286 = buf281; del buf281  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_58(c_void_p(buf285.data_ptr()), c_void_p(buf286.data_ptr()), c_void_p(relu_11.data_ptr()), c_void_p(convolution_19.data_ptr()), c_void_p(primals_220.data_ptr()), c_void_p(primals_221.data_ptr()), c_void_p(primals_55.data_ptr()), c_void_p(buf283.data_ptr()))
    del convolution_19
    del primals_220
    del primals_221
    del primals_55
    del relu_11
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf287 = aten.convolution_backward(buf286, add_35, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_35
    del primals_54
    buf288 = buf287[0]
    buf289 = buf287[1]
    del buf287
    buf293 = buf261; del buf261  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_59(c_void_p(buf256.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(primals_52.data_ptr()), c_void_p(buf293.data_ptr()))
    del primals_52
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf294 = aten.convolution_backward(buf293, mul_44, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf293
    del mul_44
    del primals_51
    buf295 = buf294[0]
    buf297 = reinterpret_tensor(buf273, (4, 120, 1, 1), (120, 1, 480, 480), 0); del buf273  # reuse
    buf298 = reinterpret_tensor(buf297, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf297  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_60(c_void_p(buf298.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(bitwise_and_6.data_ptr()))
    del bitwise_and_6
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf299 = aten.convolution_backward(buf298, relu_10, primals_49, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf298
    del primals_49
    buf300 = buf299[0]
    buf303 = buf300; del buf300  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_61(c_void_p(buf303.data_ptr()), c_void_p(relu_10.data_ptr()))
    del relu_10
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf304 = aten.convolution_backward(buf303, mean_1, primals_47, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf303
    del mean_1
    del primals_47
    buf305 = buf304[0]
    buf311 = buf286; del buf286  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_62(c_void_p(relu_9.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(primals_45.data_ptr()), c_void_p(buf311.data_ptr()))
    del primals_45
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
    buf312 = aten.convolution_backward(buf311, relu_8, primals_44, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False])
    del primals_44
    buf313 = buf312[0]
    buf318 = buf311; del buf311  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_63(c_void_p(relu_8.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(primals_42.data_ptr()), c_void_p(buf318.data_ptr()))
    del primals_42
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf319 = aten.convolution_backward(buf318, add_27, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_27
    del buf318
    del primals_41
    buf320 = buf319[0]
    buf290 = empty((40, ), device='cpu', dtype=torch.float32)
    buf291 = empty((40, ), device='cpu', dtype=torch.float32)
    buf322 = empty((40, ), device='cpu', dtype=torch.float32)
    buf323 = empty((40, ), device='cpu', dtype=torch.float32)
    buf292 = buf291; del buf291  # reuse
    cpp_fused_add_native_batch_norm_backward_64(c_void_p(buf292.data_ptr()), c_void_p(buf256.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(convolution_18.data_ptr()), c_void_p(primals_217.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(convolution_13.data_ptr()), c_void_p(primals_208.data_ptr()), c_void_p(primals_218.data_ptr()), c_void_p(buf290.data_ptr()), c_void_p(buf322.data_ptr()), c_void_p(buf323.data_ptr()))
    del convolution_13
    del convolution_18
    del primals_208
    del primals_217
    del primals_218
    buf296 = buf294[1]
    del buf294
    buf301 = buf299[1]
    buf302 = buf299[2]
    del buf299
    buf306 = buf304[1]
    buf307 = buf304[2]
    del buf304
    buf308 = empty((120, ), device='cpu', dtype=torch.float32)
    buf309 = empty((120, ), device='cpu', dtype=torch.float32)
    buf310 = buf309; del buf309  # reuse
    cpp_fused_add_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_65(c_void_p(buf310.data_ptr()), c_void_p(relu_9.data_ptr()), c_void_p(buf295.data_ptr()), c_void_p(div_2.data_ptr()), c_void_p(buf305.data_ptr()), c_void_p(convolution_15.data_ptr()), c_void_p(primals_214.data_ptr()), c_void_p(primals_215.data_ptr()), c_void_p(buf308.data_ptr()))
    del buf295
    del buf305
    del convolution_15
    del div_2
    del primals_214
    del primals_215
    del relu_9
    buf314 = buf312[1]
    del buf312
    buf315 = empty((120, ), device='cpu', dtype=torch.float32)
    buf316 = empty((120, ), device='cpu', dtype=torch.float32)
    buf317 = buf316; del buf316  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_66(c_void_p(buf317.data_ptr()), c_void_p(relu_8.data_ptr()), c_void_p(buf313.data_ptr()), c_void_p(convolution_14.data_ptr()), c_void_p(primals_211.data_ptr()), c_void_p(primals_212.data_ptr()), c_void_p(buf315.data_ptr()))
    del buf313
    del convolution_14
    del primals_211
    del primals_212
    del relu_8
    buf321 = buf319[1]
    del buf319
    buf324 = buf323; del buf323  # reuse
    buf325 = buf256; del buf256  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_67(c_void_p(buf324.data_ptr()), c_void_p(buf325.data_ptr()), c_void_p(primals_209.data_ptr()), c_void_p(buf288.data_ptr()), c_void_p(buf320.data_ptr()), c_void_p(primals_39.data_ptr()))
    del buf288
    del buf320
    del primals_209
    del primals_39
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf326 = aten.convolution_backward(buf325, mul_34, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf325
    del mul_34
    del primals_38
    buf327 = buf326[0]
    buf328 = buf326[1]
    del buf326
    buf329 = empty_strided((4, 72, 1, 1), (72, 1, 288, 288), device='cpu', dtype=torch.float32)
    buf330 = reinterpret_tensor(buf329, (4, 72, 1, 1), (72, 1, 72, 72), 0); del buf329  # reuse
    cpp_fused_convolution_backward_hardsigmoid_backward_hardswish_backward_mul_sum_68(c_void_p(buf330.data_ptr()), c_void_p(buf327.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(bitwise_and_7.data_ptr()))
    del bitwise_and_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardsigmoid_backward, aten.hardswish_backward]
    buf331 = aten.convolution_backward(buf330, relu_7, primals_36, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf330
    del primals_36
    buf332 = buf331[0]
    buf333 = buf331[1]
    buf334 = buf331[2]
    del buf331
    buf335 = buf332; del buf332  # reuse
    cpp_fused_convolution_backward_hardswish_backward_threshold_backward_69(c_void_p(buf335.data_ptr()), c_void_p(relu_7.data_ptr()))
    del relu_7
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.threshold_backward]
    buf336 = aten.convolution_backward(buf335, mean, primals_34, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True])
    del buf335
    del mean
    del primals_34
    buf337 = buf336[0]
    buf338 = buf336[1]
    buf339 = buf336[2]
    del buf336
    buf340 = empty((72, ), device='cpu', dtype=torch.float32)
    buf341 = empty((72, ), device='cpu', dtype=torch.float32)
    buf342 = buf341; del buf341  # reuse
    buf343 = buf327; del buf327  # reuse
    cpp_fused_add_convolution_backward_div_hardswish_backward_mul_native_batch_norm_backward_threshold_backward_70(c_void_p(buf342.data_ptr()), c_void_p(buf343.data_ptr()), c_void_p(relu_6.data_ptr()), c_void_p(div_1.data_ptr()), c_void_p(buf337.data_ptr()), c_void_p(convolution_10.data_ptr()), c_void_p(primals_205.data_ptr()), c_void_p(primals_206.data_ptr()), c_void_p(primals_32.data_ptr()), c_void_p(buf340.data_ptr()))
    del buf337
    del convolution_10
    del div_1
    del primals_205
    del primals_206
    del primals_32
    del relu_6
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.div, aten.hardswish_backward, aten.mul, aten.native_batch_norm_backward, aten.threshold_backward]
    buf344 = aten.convolution_backward(buf343, relu_5, primals_31, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False])
    del buf343
    del primals_31
    buf345 = buf344[0]
    buf346 = buf344[1]
    del buf344
    buf347 = empty((72, ), device='cpu', dtype=torch.float32)
    buf348 = empty((72, ), device='cpu', dtype=torch.float32)
    buf349 = buf348; del buf348  # reuse
    buf350 = buf345; del buf345  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_71(c_void_p(buf349.data_ptr()), c_void_p(buf350.data_ptr()), c_void_p(relu_5.data_ptr()), c_void_p(convolution_9.data_ptr()), c_void_p(primals_202.data_ptr()), c_void_p(primals_203.data_ptr()), c_void_p(primals_29.data_ptr()), c_void_p(buf347.data_ptr()))
    del convolution_9
    del primals_202
    del primals_203
    del primals_29
    del relu_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf351 = aten.convolution_backward(buf350, add_20, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_20
    del primals_28
    buf352 = buf351[0]
    buf353 = buf351[1]
    del buf351
    buf357 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    cpp_fused_convolution_backward_native_batch_norm_backward_72(c_void_p(buf352.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(primals_26.data_ptr()), c_void_p(buf357.data_ptr()))
    del primals_26
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf358 = aten.convolution_backward(buf357, relu_4, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf357
    del primals_25
    buf359 = buf358[0]
    buf364 = buf350; del buf350  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_73(c_void_p(relu_4.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(primals_23.data_ptr()), c_void_p(buf364.data_ptr()))
    del primals_23
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf365 = aten.convolution_backward(buf364, relu_3, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False])
    del primals_22
    buf366 = buf365[0]
    buf371 = buf364; del buf364  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_74(c_void_p(relu_3.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(primals_20.data_ptr()), c_void_p(buf371.data_ptr()))
    del primals_20
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf372 = aten.convolution_backward(buf371, add_13, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_13
    del buf371
    del primals_19
    buf373 = buf372[0]
    buf354 = empty((24, ), device='cpu', dtype=torch.float32)
    buf355 = empty((24, ), device='cpu', dtype=torch.float32)
    buf375 = empty((24, ), device='cpu', dtype=torch.float32)
    buf376 = empty((24, ), device='cpu', dtype=torch.float32)
    buf356 = buf355; del buf355  # reuse
    cpp_fused_add_native_batch_norm_backward_75(c_void_p(buf356.data_ptr()), c_void_p(buf352.data_ptr()), c_void_p(convolution_8.data_ptr()), c_void_p(primals_199.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(convolution_5.data_ptr()), c_void_p(primals_190.data_ptr()), c_void_p(primals_200.data_ptr()), c_void_p(buf354.data_ptr()), c_void_p(buf375.data_ptr()), c_void_p(buf376.data_ptr()))
    del convolution_5
    del convolution_8
    del primals_190
    del primals_199
    del primals_200
    buf360 = buf358[1]
    del buf358
    buf361 = empty((72, ), device='cpu', dtype=torch.float32)
    buf362 = empty((72, ), device='cpu', dtype=torch.float32)
    buf363 = buf362; del buf362  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_76(c_void_p(buf363.data_ptr()), c_void_p(relu_4.data_ptr()), c_void_p(buf359.data_ptr()), c_void_p(convolution_7.data_ptr()), c_void_p(primals_196.data_ptr()), c_void_p(primals_197.data_ptr()), c_void_p(buf361.data_ptr()))
    del buf359
    del convolution_7
    del primals_196
    del primals_197
    del relu_4
    buf367 = buf365[1]
    del buf365
    buf368 = empty((72, ), device='cpu', dtype=torch.float32)
    buf369 = empty((72, ), device='cpu', dtype=torch.float32)
    buf370 = buf369; del buf369  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_77(c_void_p(buf370.data_ptr()), c_void_p(relu_3.data_ptr()), c_void_p(buf366.data_ptr()), c_void_p(convolution_6.data_ptr()), c_void_p(primals_193.data_ptr()), c_void_p(primals_194.data_ptr()), c_void_p(buf368.data_ptr()))
    del buf366
    del convolution_6
    del primals_193
    del primals_194
    del relu_3
    buf374 = buf372[1]
    del buf372
    buf377 = buf376; del buf376  # reuse
    buf378 = buf352; del buf352  # reuse
    cpp_fused_add_convolution_backward_native_batch_norm_backward_78(c_void_p(buf377.data_ptr()), c_void_p(buf378.data_ptr()), c_void_p(primals_191.data_ptr()), c_void_p(buf373.data_ptr()), c_void_p(primals_17.data_ptr()))
    del buf373
    del primals_17
    del primals_191
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.native_batch_norm_backward]
    buf379 = aten.convolution_backward(buf378, relu_2, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del buf378
    del primals_16
    buf380 = buf379[0]
    buf381 = buf379[1]
    del buf379
    buf382 = empty((64, ), device='cpu', dtype=torch.float32)
    buf383 = empty((64, ), device='cpu', dtype=torch.float32)
    buf384 = buf383; del buf383  # reuse
    buf385 = buf380; del buf380  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_79(c_void_p(buf384.data_ptr()), c_void_p(buf385.data_ptr()), c_void_p(relu_2.data_ptr()), c_void_p(convolution_4.data_ptr()), c_void_p(primals_187.data_ptr()), c_void_p(primals_188.data_ptr()), c_void_p(primals_14.data_ptr()), c_void_p(buf382.data_ptr()))
    del convolution_4
    del primals_14
    del primals_187
    del primals_188
    del relu_2
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf386 = aten.convolution_backward(buf385, relu_1, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False])
    del primals_13
    buf387 = buf386[0]
    buf388 = buf386[1]
    del buf386
    buf389 = empty((64, ), device='cpu', dtype=torch.float32)
    buf390 = empty((64, ), device='cpu', dtype=torch.float32)
    buf391 = buf390; del buf390  # reuse
    buf392 = buf387; del buf387  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_80(c_void_p(buf391.data_ptr()), c_void_p(buf392.data_ptr()), c_void_p(relu_1.data_ptr()), c_void_p(convolution_3.data_ptr()), c_void_p(primals_184.data_ptr()), c_void_p(primals_185.data_ptr()), c_void_p(primals_11.data_ptr()), c_void_p(buf389.data_ptr()))
    del convolution_3
    del primals_11
    del primals_184
    del primals_185
    del relu_1
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf393 = aten.convolution_backward(buf392, add_7, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del add_7
    del buf392
    del primals_10
    buf394 = buf393[0]
    buf395 = buf393[1]
    del buf393
    buf399 = reinterpret_tensor(buf385, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf385  # reuse
    cpp_fused_convolution_backward_native_batch_norm_backward_81(c_void_p(buf394.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(primals_8.data_ptr()), c_void_p(buf399.data_ptr()))
    del primals_8
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.native_batch_norm_backward]
    buf400 = aten.convolution_backward(buf399, relu, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False])
    del primals_7
    buf401 = buf400[0]
    buf406 = buf399; del buf399  # reuse
    cpp_fused_convolution_backward_hardswish_backward_native_batch_norm_backward_threshold_backward_82(c_void_p(relu.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(primals_5.data_ptr()), c_void_p(buf406.data_ptr()))
    del primals_5
    # Source Nodes: [], Original ATen: [aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward, aten.threshold_backward]
    buf407 = aten.convolution_backward(buf406, div, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False])
    del buf406
    del div
    del primals_4
    buf408 = buf407[0]
    buf396 = empty((16, ), device='cpu', dtype=torch.float32)
    buf397 = empty((16, ), device='cpu', dtype=torch.float32)
    buf410 = empty((16, ), device='cpu', dtype=torch.float32)
    buf411 = empty((16, ), device='cpu', dtype=torch.float32)
    buf398 = buf397; del buf397  # reuse
    cpp_fused_add_hardswish_backward_native_batch_norm_backward_83(c_void_p(buf398.data_ptr()), c_void_p(buf394.data_ptr()), c_void_p(convolution_2.data_ptr()), c_void_p(primals_181.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(convolution.data_ptr()), c_void_p(primals_175.data_ptr()), c_void_p(primals_182.data_ptr()), c_void_p(buf396.data_ptr()), c_void_p(buf410.data_ptr()), c_void_p(buf411.data_ptr()))
    del convolution
    del convolution_2
    del primals_175
    del primals_181
    del primals_182
    buf402 = buf400[1]
    del buf400
    buf403 = empty((16, ), device='cpu', dtype=torch.float32)
    buf404 = empty((16, ), device='cpu', dtype=torch.float32)
    buf405 = buf404; del buf404  # reuse
    cpp_fused_hardswish_backward_native_batch_norm_backward_threshold_backward_84(c_void_p(buf405.data_ptr()), c_void_p(relu.data_ptr()), c_void_p(buf401.data_ptr()), c_void_p(convolution_1.data_ptr()), c_void_p(primals_178.data_ptr()), c_void_p(primals_179.data_ptr()), c_void_p(buf403.data_ptr()))
    del buf401
    del convolution_1
    del primals_178
    del primals_179
    del relu
    buf409 = buf407[1]
    del buf407
    buf412 = buf411; del buf411  # reuse
    buf413 = buf394; del buf394  # reuse
    cpp_fused_add_convolution_backward_hardswish_backward_native_batch_norm_backward_85(c_void_p(buf412.data_ptr()), c_void_p(buf413.data_ptr()), c_void_p(primals_176.data_ptr()), c_void_p(clone.data_ptr()), c_void_p(buf408.data_ptr()), c_void_p(primals_2.data_ptr()))
    del buf408
    del clone
    del primals_176
    del primals_2
    # Source Nodes: [], Original ATen: [aten.add, aten.convolution_backward, aten.hardswish_backward, aten.native_batch_norm_backward]
    buf414 = aten.convolution_backward(buf413, primals_313, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False])
    del buf413
    del primals_1
    del primals_313
    buf415 = buf414[1]
    return (buf415, buf412, buf410, buf409, buf405, buf403, buf402, buf398, buf396, buf395, buf391, buf389, buf388, buf384, buf382, buf381, buf377, buf375, buf374, buf370, buf368, buf367, buf363, buf361, buf360, buf356, buf354, buf353, buf349, buf347, buf346, buf342, buf340, buf338, buf339, buf333, buf334, buf328, buf324, buf322, buf321, buf317, buf315, buf314, buf310, buf308, buf306, buf307, buf301, buf302, buf296, buf292, buf290, buf289, buf285, buf283, buf282, buf278, buf276, buf274, buf275, buf269, buf270, buf264, buf260, buf258, buf257, buf253, buf251, buf250, buf246, buf244, buf243, buf239, buf237, buf236, buf232, buf230, buf229, buf225, buf223, buf222, buf218, buf216, buf215, buf211, buf209, buf208, buf204, buf202, buf201, buf197, buf195, buf194, buf190, buf188, buf187, buf183, buf181, buf180, buf176, buf174, buf173, buf169, buf167, buf166, buf162, buf160, buf158, buf159, buf153, buf154, buf148, buf144, buf142, buf141, buf137, buf135, buf134, buf130, buf128, buf126, buf127, buf121, buf122, buf116, buf112, buf110, buf109, buf105, buf103, buf102, buf98, buf96, buf94, buf95, buf89, buf90, buf84, buf80, buf78, buf77, buf73, buf71, buf70, buf66, buf64, buf62, buf63, buf57, buf58, buf52, buf48, buf46, buf45, buf41, buf39, buf38, buf34, buf32, buf30, buf31, buf25, buf26, buf20, buf16, buf14, buf13, buf9, buf7, reinterpret_tensor(buf5, (1280, 960), (960, 1), 0), buf6, reinterpret_tensor(buf1, (1000, 1280), (1280, 1), 0), buf2, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cpu', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_4 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_7 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_10 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_13 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_16 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_17 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_19 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_20 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_22 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_23 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_25 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_28 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_29 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_31 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_32 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_34 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_36 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_38 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_39 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_41 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_42 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_44 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_45 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_47 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_49 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_51 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_52 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_54 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_55 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_57 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_58 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_60 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_62 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_64 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_65 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_67 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_68 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_70 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_71 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_73 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_74 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_76 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_77 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_79 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_80 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_82 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_83 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_85 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_86 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_88 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_89 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_91 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_92 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_94 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_95 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_97 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_98 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_100 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_101 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_103 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_104 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_106 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_107 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_109 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_111 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_113 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_114 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_116 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_117 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_119 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cpu', dtype=torch.float32)
    primals_120 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_122 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_124 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_126 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_127 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_129 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_130 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_132 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_133 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_135 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_137 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_139 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_142 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_143 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_145 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_146 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_148 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_150 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_152 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_153 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_155 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_156 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_158 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cpu', dtype=torch.float32)
    primals_159 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_161 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_163 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_165 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_166 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_168 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cpu', dtype=torch.float32)
    primals_169 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_175 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_176 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_178 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_179 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_181 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_182 = rand_strided((16, ), (1, ), device='cpu', dtype=torch.float32)
    primals_184 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_185 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_187 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_188 = rand_strided((64, ), (1, ), device='cpu', dtype=torch.float32)
    primals_190 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_191 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_193 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_194 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_196 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_197 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_199 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_200 = rand_strided((24, ), (1, ), device='cpu', dtype=torch.float32)
    primals_202 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_203 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_205 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_206 = rand_strided((72, ), (1, ), device='cpu', dtype=torch.float32)
    primals_208 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_209 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_211 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_212 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_214 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_215 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_217 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_218 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_220 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_221 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_223 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_224 = rand_strided((120, ), (1, ), device='cpu', dtype=torch.float32)
    primals_226 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_227 = rand_strided((40, ), (1, ), device='cpu', dtype=torch.float32)
    primals_229 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_230 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_232 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_233 = rand_strided((240, ), (1, ), device='cpu', dtype=torch.float32)
    primals_235 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_236 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_238 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_239 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_241 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_242 = rand_strided((200, ), (1, ), device='cpu', dtype=torch.float32)
    primals_244 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_245 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_247 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_248 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_250 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_251 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_253 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_254 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_256 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_257 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_259 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_260 = rand_strided((184, ), (1, ), device='cpu', dtype=torch.float32)
    primals_262 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_263 = rand_strided((80, ), (1, ), device='cpu', dtype=torch.float32)
    primals_265 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_266 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_268 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_269 = rand_strided((480, ), (1, ), device='cpu', dtype=torch.float32)
    primals_271 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_272 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_274 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_275 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_277 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_278 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_280 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_281 = rand_strided((112, ), (1, ), device='cpu', dtype=torch.float32)
    primals_283 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_284 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_286 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_287 = rand_strided((672, ), (1, ), device='cpu', dtype=torch.float32)
    primals_289 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_290 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_292 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_293 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_295 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_296 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_298 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_299 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_301 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_302 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_304 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_305 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_307 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_308 = rand_strided((160, ), (1, ), device='cpu', dtype=torch.float32)
    primals_310 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_311 = rand_strided((960, ), (1, ), device='cpu', dtype=torch.float32)
    primals_313 = rand_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cpu', dtype=torch.float32)
    convolution = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    clone = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    div = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_1 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    relu = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_2 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    add_7 = rand_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cpu', dtype=torch.float32)
    convolution_3 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    relu_1 = rand_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cpu', dtype=torch.float32)
    convolution_4 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    relu_2 = rand_strided((4, 64, 56, 56), (200704, 1, 3584, 64), device='cpu', dtype=torch.float32)
    convolution_5 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_13 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_6 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_3 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_7 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_4 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_8 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    add_20 = rand_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cpu', dtype=torch.float32)
    convolution_9 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    relu_5 = rand_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cpu', dtype=torch.float32)
    convolution_10 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    relu_6 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    mean = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    relu_7 = rand_strided((4, 24, 1, 1), (24, 1, 24, 24), device='cpu', dtype=torch.float32)
    div_1 = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.float32)
    mul_34 = rand_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cpu', dtype=torch.float32)
    convolution_13 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_27 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_14 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_8 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_15 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_9 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_1 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    relu_10 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_2 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_44 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_18 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_35 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_19 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_11 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_20 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    relu_12 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    mean_2 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    relu_13 = rand_strided((4, 32, 1, 1), (32, 1, 32, 32), device='cpu', dtype=torch.float32)
    div_3 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    mul_54 = rand_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cpu', dtype=torch.float32)
    convolution_23 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    add_43 = rand_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cpu', dtype=torch.float32)
    convolution_24 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    clone_1 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    div_4 = rand_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cpu', dtype=torch.float32)
    convolution_25 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    clone_2 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    div_5 = rand_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cpu', dtype=torch.float32)
    convolution_26 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_51 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_27 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    clone_3 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    div_6 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_28 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    clone_4 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    div_7 = rand_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cpu', dtype=torch.float32)
    convolution_29 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_60 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_30 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    clone_5 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_8 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_31 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    clone_6 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_9 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_32 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_69 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_33 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    clone_7 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_10 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_34 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    clone_8 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    div_11 = rand_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cpu', dtype=torch.float32)
    convolution_35 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    add_78 = rand_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cpu', dtype=torch.float32)
    convolution_36 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    clone_9 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    div_12 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_37 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    clone_10 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    div_13 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    mean_3 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    relu_14 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.float32)
    div_14 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.float32)
    mul_110 = rand_strided((4, 480, 14, 14), (94080, 1, 6720, 480), device='cpu', dtype=torch.float32)
    convolution_40 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    add_87 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_41 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    clone_11 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_15 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_42 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    clone_12 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_16 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    mean_4 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_15 = rand_strided((4, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_17 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_122 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_45 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    add_97 = rand_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cpu', dtype=torch.float32)
    convolution_46 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    clone_13 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    div_18 = rand_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cpu', dtype=torch.float32)
    convolution_47 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    clone_14 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    div_19 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    mean_5 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    relu_16 = rand_strided((4, 168, 1, 1), (168, 1, 168, 168), device='cpu', dtype=torch.float32)
    div_20 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.float32)
    mul_134 = rand_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cpu', dtype=torch.float32)
    convolution_50 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    add_106 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_51 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    clone_15 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_21 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_52 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    clone_16 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_22 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_6 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_17 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_23 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_146 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_55 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    add_116 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_56 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    clone_17 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_24 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_57 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    clone_18 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    div_25 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    mean_7 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    relu_18 = rand_strided((4, 240, 1, 1), (240, 1, 240, 240), device='cpu', dtype=torch.float32)
    div_26 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.float32)
    mul_158 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    convolution_60 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    add_126 = rand_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cpu', dtype=torch.float32)
    convolution_61 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    clone_19 = rand_strided((4, 960, 7, 7), (47040, 1, 6720, 960), device='cpu', dtype=torch.float32)
    view = rand_strided((4, 960), (960, 1), device='cpu', dtype=torch.float32)
    addmm = rand_strided((4, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    div_28 = rand_strided((4, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_2 = rand_strided((1000, 1280), (1280, 1), device='cpu', dtype=torch.float32)
    permute_6 = rand_strided((1280, 960), (960, 1), device='cpu', dtype=torch.float32)
    bitwise_and = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    bitwise_and_1 = rand_strided((4, 960, 1, 1), (960, 1, 960, 960), device='cpu', dtype=torch.bool)
    bitwise_and_2 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    bitwise_and_3 = rand_strided((4, 672, 1, 1), (672, 1, 672, 672), device='cpu', dtype=torch.bool)
    bitwise_and_4 = rand_strided((4, 480, 1, 1), (480, 1, 480, 480), device='cpu', dtype=torch.bool)
    bitwise_and_5 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    bitwise_and_6 = rand_strided((4, 120, 1, 1), (120, 1, 120, 120), device='cpu', dtype=torch.bool)
    bitwise_and_7 = rand_strided((4, 72, 1, 1), (72, 1, 72, 72), device='cpu', dtype=torch.bool)
    tangents_1 = rand_strided((4, 1000), (1000, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_47, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_111, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_150, primals_152, primals_153, primals_155, primals_156, primals_158, primals_159, primals_161, primals_163, primals_165, primals_166, primals_168, primals_169, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_313, convolution, clone, div, convolution_1, relu, convolution_2, add_7, convolution_3, relu_1, convolution_4, relu_2, convolution_5, add_13, convolution_6, relu_3, convolution_7, relu_4, convolution_8, add_20, convolution_9, relu_5, convolution_10, relu_6, mean, relu_7, div_1, mul_34, convolution_13, add_27, convolution_14, relu_8, convolution_15, relu_9, mean_1, relu_10, div_2, mul_44, convolution_18, add_35, convolution_19, relu_11, convolution_20, relu_12, mean_2, relu_13, div_3, mul_54, convolution_23, add_43, convolution_24, clone_1, div_4, convolution_25, clone_2, div_5, convolution_26, add_51, convolution_27, clone_3, div_6, convolution_28, clone_4, div_7, convolution_29, add_60, convolution_30, clone_5, div_8, convolution_31, clone_6, div_9, convolution_32, add_69, convolution_33, clone_7, div_10, convolution_34, clone_8, div_11, convolution_35, add_78, convolution_36, clone_9, div_12, convolution_37, clone_10, div_13, mean_3, relu_14, div_14, mul_110, convolution_40, add_87, convolution_41, clone_11, div_15, convolution_42, clone_12, div_16, mean_4, relu_15, div_17, mul_122, convolution_45, add_97, convolution_46, clone_13, div_18, convolution_47, clone_14, div_19, mean_5, relu_16, div_20, mul_134, convolution_50, add_106, convolution_51, clone_15, div_21, convolution_52, clone_16, div_22, mean_6, relu_17, div_23, mul_146, convolution_55, add_116, convolution_56, clone_17, div_24, convolution_57, clone_18, div_25, mean_7, relu_18, div_26, mul_158, convolution_60, add_126, convolution_61, clone_19, view, addmm, div_28, permute_2, permute_6, bitwise_and, bitwise_and_1, bitwise_and_2, bitwise_and_3, bitwise_and_4, bitwise_and_5, bitwise_and_6, bitwise_and_7, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v3_large', benchmark_compiled_module)
